import os
import os.path as osp
import cv2
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
#os.chdir(CODE_SPACE)
import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mmcv.utils import Config, DictAction
from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.database import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_data_rgb_depth_intrinsic_norm, load_from_annos

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm', help='job launcher')
    parser.add_argument('--test_anno_path', default=None, type=str, help='the path of test data')
    parser.add_argument('--data_root', type=str, help='the data_root of test annotation path', default='')
    parser.add_argument('--in_the_wild_rgb_folder', default=None, type=str, help='the path of test data')
    args = parser.parse_args()
    return args

def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        
    # show_dir is determined in this priority: CLI > segment in file > filename
    if args.show_dir is not None:
        # update configs according to CLI args if args.show_dir is not None
        cfg.show_dir = args.show_dir
    else:
        # use condig filename + timestamp as default show_dir if args.show_dir is None
        cfg.show_dir = osp.join('./show_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp)
    
    # ckpt path
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    
    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.data_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)
    
    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)
    
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # init distributed env dirst, since logger depends on the dist info
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')
    
    # dump config 
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    test_anno_path = args.test_anno_path
    in_the_wild_rgb_folder = args.in_the_wild_rgb_folder

    if (test_anno_path is not None) and (args.data_root is not None):
        if not os.path.isabs(test_anno_path):
            test_anno_path = osp.join(CODE_SPACE, test_anno_path)
        test_data = load_from_annos(test_anno_path, args.data_root)
    elif in_the_wild_rgb_folder is not None:
        test_data = []
        for rgb_name in sorted(os.listdir(in_the_wild_rgb_folder)):
            rgb = osp.join(in_the_wild_rgb_folder, rgb_name)
            image = cv2.imread(rgb)
            test_data_i = {
                'rgb': rgb,
                'depth': None,
                'depth_mask': None,
                'norm': None,
                'depth_scale': 1.0,
                'intrinsic': [1000, 1000, image.shape[0]/2, image.shape[1]/2],
                'filename': os.path.basename(rgb),
                'folder': rgb.split('/')[-2],
            }
            test_data.append(test_data_i)
    else:
        raise ValueError("Input args of either --test_anno_path or --in_the_wild_rgb_folder")
    
    """
    # NOTE: you can also change rgb_paths, intrinsic, depth_paths, and depth_scale here straightforwardly. Here is an example of scannet
    base_root = '/mnt/nas/share/home/xugk/data/scannet_test/scannet_test'
    depth_scale = 1000
    intrinsic = [1165.723022, 1165.738037, 649.094971, 484.765015]
    scene_names = ['scene07%02d_00' % i for i in range(7, 10)]
    for scene_name in scene_names:
        rgb_root = osp.join(base_root, scene_name, 'color')
        for rgb_path in os.listdir(rgb_root):
            rgb_path = osp.join(rgb_root, rgb_path)
            rgb_paths.append(rgb_path)
            depth_path = rgb_path.replace('/color/', '/depth/').replace('.jpg', '.png')
            depth_paths.append(depth_path)
    test_data = load_data_rgb_depth_intrinsic_norm(rgb_paths, intrinsic, depths=depth_paths, depth_scale=depth_scale)
    """
    
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher, test_data)
    else:
        # distributed training
        if args.launcher == 'ror':
            local_rank = cfg.dist_params.local_rank
            main_worker(local_rank, cfg, args.launcher, test_data)
        else:
            mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher, test_data))
        
def main_worker(local_rank: int, cfg: dict, launcher: str, test_data: list):
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank

        if launcher == 'ror':
            init_torch_process_group(use_hvd=False)
        else:
            torch.cuda.set_device(local_rank)
            default_timeout = timedelta(minutes=30)
            dist.init_process_group(
                backend=cfg.dist_params.backend,
                init_method=cfg.dist_params.dist_url,
                world_size=cfg.dist_params.world_size,
                rank=cfg.dist_params.global_rank,
                timeout=default_timeout)
    
    logger = setup_logger(cfg.log_file)
    # build model
    model = get_configured_monodepth_model(cfg, None, )
    
    # config distributed training
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    # load ckpt
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=True)
    model.eval()
    
    do_scalecano_test_with_custom_data(
        model, 
        cfg,
        test_data,
        logger,
        cfg.distributed,
        local_rank
    )
    
if __name__ == '__main__':
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    if args.launcher == 'ror':
        from ac2.ror import DistributedTorchCarrier
        from ac2.ror.integration.mmdetection import init_torch_process_group
        DistributedTorchCarrier(set_device_from_local_rank=True)(main)(args)
    else:
        main(args)