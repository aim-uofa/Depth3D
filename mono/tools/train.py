import os
import os.path as osp
import time
import sys
CODE_SPACE=osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(CODE_SPACE)
import argparse
import copy
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mmcv.utils import Config, DictAction
import socket
import subprocess
from datetime import timedelta
import random
import numpy as np
import logging

from mono.datasets.distributed_sampler import log_canonical_transfer_info
from mono.utils.comm import init_env
from mono.utils.logger import setup_logger
from mono.utils.database import load_data_info, reset_ckpt_path
from mono.utils.do_train import do_train




def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--tensorboard-dir', help='the dir to save tensorboard logs')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--use-tensorboard',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', 
                        type=int, 
                        default=1,
                        help='number of nodes')
    parser.add_argument(
        '--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm',
        help='job launcher')

    args = parser.parse_args()
    return args

  
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        @seed (int): Seed to be used.
        @deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if deterministic:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    cfg.dist_params.nnodes = args.nnodes
    cfg.dist_params.node_rank = args.node_rank

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename + timestamp as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp)
    # tensorboard_dir is determined in this priority: CLI > segment in file > filename
    if args.tensorboard_dir is not None:
        cfg.tensorboard_dir = args.tensorboard_dir
    elif cfg.get('tensorboard_dir', None) is None:
        # use cfg.work_dir + 'tensorboard' as default tensorboard_dir if cfg.tensorboard_dir is None
        cfg.tensorboard_dir = osp.join(cfg.work_dir, 'tensorboard')
    
    # ckpt path
    if args.load_from is not None:
        cfg.load_from = args.load_from
    # resume training
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # create work_dir and tensorboard_dir
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    os.makedirs(osp.abspath(cfg.tensorboard_dir), exist_ok=True)

    # init the logger before other steps
    cfg.log_file = osp.join(cfg.work_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # # log env info
    # env_info_dict = collect_env()
    # env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    # meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # mute online evaluation
    if args.no_validate:
        cfg.evaluation.online_eval = False

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.data_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)

    # log data transfer to canonical space info
    # log_canonical_transfer_info(cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
    init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    if not cfg.distributed:
        main_worker(0, cfg)
    else:
        # distributed training
        if args.launcher == 'ror':
            local_rank = cfg.dist_params.local_rank
            main_worker(local_rank, cfg, args.launcher)
        else:
            mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher))
        
 
def main_worker(local_rank: int, cfg: dict, launcher: str='slurm'):
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank
        os.environ['RANK']=str(cfg.dist_params.global_rank)

        if launcher == 'ror':
            init_torch_process_group(use_hvd=False)
        else:
            torch.cuda.set_device(local_rank)
            default_timeout = timedelta(minutes=30)
            dist.init_process_group(
                backend=cfg.dist_params.backend,
                init_method=cfg.dist_params.dist_url,
                world_size=cfg.dist_params.world_size,
                rank=cfg.dist_params.global_rank)
    
    set_random_seed(cfg.seed)
    do_train(local_rank, cfg)
        

if __name__=='__main__':
    # load args
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    print(args.work_dir, args.tensorboard_dir)
    if args.launcher == 'ror':
        from ac2.ror import DistributedTorchCarrier
        from ac2.ror.integration.mmdetection import init_torch_process_group
        DistributedTorchCarrier(set_device_from_local_rank=True)(main)(args)
    else:
        main(args)