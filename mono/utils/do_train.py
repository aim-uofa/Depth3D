import os
import torch
import matplotlib.pyplot as plt
from mono.model.monodepth_model import get_configured_monodepth_model
from tensorboardX import SummaryWriter
from mono.utils.comm import TrainingStats
from mono.utils.avg_meter import MetricAverageMeter, NormalMeter
from mono.utils.running import build_lr_schedule_with_cfg, build_optimizer_with_cfg, load_ckpt, save_ckpt
from mono.utils.comm import reduce_dict, main_process
from mono.utils.visualization import save_val_imgs, visual_train_data, create_html
import traceback
from mono.utils.visualization import create_dir_for_validate_meta
from mono.model.criterion import build_criterions
from mono.datasets.distributed_sampler import build_dataset_n_sampler_with_cfg, build_sampler_with_dataset_cfg
from mono.utils.logger import setup_logger
import logging
from .misc import NativeScalerWithGradNormCount, is_bf16_supported
import math
import sys
import random
import numpy as np

if str(torch.__version__).startswith('2.'):
    pytorch_2_0 = True
else:
    pytorch_2_0 = False

# if pytorch_2_0:
#     import torch._dynamo
#     torch._dynamo.config.suppress_errors = True

def to_cuda(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale

def do_train(local_rank: int, cfg: dict):

    logger = setup_logger(cfg.log_file)

    # build criterions
    criterions = build_criterions(cfg)
    
    # build model
    model = get_configured_monodepth_model(cfg,
                                           criterions,
                                           )
    
    # for name, param in model.named_parameters():
    #     print(name)
    # assert False
    
    if 'dinov2_' in cfg.model.backbone.type and cfg.model.backbone.freeze_backbone==True:
        for name, param in model.depth_model.encoder.named_parameters():
            if 'dinov2_out_adapter' not in name:
                # print('False :', name)
                param.requires_grad = False
            else:
                # print('True :', name)
                pass
    
    # log model state_dict
    if main_process():
        logger.info(model.state_dict().keys())
    
    # build datasets
    train_dataset, train_sampler = build_dataset_n_sampler_with_cfg(cfg, 'train')
    val_dataset, val_sampler = build_dataset_n_sampler_with_cfg(cfg, 'val')
    # build data loaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.batchsize_per_gpu,
                                                   num_workers=cfg.thread_per_gpu,
                                                   sampler=train_sampler,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   prefetch_factor=2,)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=1,
                                                 num_workers=1,
                                                 sampler=val_sampler,
                                                 drop_last=True,
                                                 pin_memory=True,
                                                 prefetch_factor=2,)

    
    # build schedule
    lr_scheduler = build_lr_schedule_with_cfg(cfg)
    optimizer = build_optimizer_with_cfg(cfg, model)
    # assert False
   
    # config distributed training
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), 
                                                          device_ids=[local_rank], 
                                                          output_device=local_rank, 
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())
    
    # init automatic mix precision training
    if 'AMP' in cfg.runner.type:
        loss_scaler = NativeScalerWithGradNormCount()
    else:
        loss_scaler = None
    
    train_dataloader_resume = None
    # load ckpt
    if cfg.load_from and cfg.resume_from is None:
        model, _, _, loss_scaler = load_ckpt(cfg.load_from, model, optimizer=None, scheduler=None, strict_match=False, loss_scaler=loss_scaler)
    elif cfg.resume_from:
        model, optimizer, lr_scheduler, loss_scaler = load_ckpt(
            cfg.resume_from, 
            model, 
            optimizer=optimizer, 
            scheduler=lr_scheduler, 
            strict_match=True, 
            loss_scaler=loss_scaler)
        
        # dataloader for the first epoch of resume training.
        resume_offset = lr_scheduler.state_dict()['_step_count'] * cfg.batchsize_per_gpu
        train_sampler_resume = build_sampler_with_dataset_cfg(cfg, train_dataset, 'train', resume_offset=resume_offset)

        train_dataloader_resume = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=cfg.batchsize_per_gpu,
                                                    num_workers=cfg.thread_per_gpu,
                                                    sampler=train_sampler_resume,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    prefetch_factor=2,)
    # # torch.compile only for pytorch >= 2.0.0
    # if pytorch_2_0:
    #     model.module.depth_model = torch.compile(model.module.depth_model)

    if cfg.runner.type == 'IterBasedRunner':
        train_by_iters(cfg,
                    model, 
                    optimizer, 
                    lr_scheduler,
                    train_dataloader,
                    val_dataloader,
                    train_dataloader_resume=train_dataloader_resume,
                    )
    elif cfg.runner.type == 'IterBasedRunner_MultiSize':
        train_by_iters_multisize(
            cfg = cfg,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
    elif cfg.runner.type == 'IterBasedRunner_AMP':
        train_by_iters_amp(
            cfg = cfg,
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_scaler=loss_scaler
        )
    elif cfg.runner.type == 'EpochBasedRunner':
        raise RuntimeError('It is not supported currently. :)')
    else:
        raise RuntimeError('It is not supported currently. :)')


def train_by_iters(cfg, model, optimizer, lr_scheduler, train_dataloader, val_dataloader, train_dataloader_resume=None):
    """
    Do the training by iterations.
    """
    logger = logging.getLogger()
    tb_logger = None
    if cfg.use_tensorboard and main_process():
        tb_logger = SummaryWriter(cfg.tensorboard_dir)
    if main_process():
        training_stats = TrainingStats(log_period=cfg.log_interval, tensorboard_logger=tb_logger)
    
    lr_scheduler.before_run(optimizer)
    
    # set training steps
    max_iters = cfg.runner.max_iters
    start_iter = lr_scheduler._step_count

    save_interval = cfg.checkpoint_config.interval
    eval_interval = cfg.evaluation.interval
    epoch = 0
    logger.info('Create iterator.')
    if train_dataloader_resume is not None:
        dataloader_iterator = iter(train_dataloader_resume)
    else:
        dataloader_iterator = iter(train_dataloader)
    
    logger.info('Training dataloader length : ' + str(len(dataloader_iterator)))

    val_err = {}
    logger.info('Start training.')
    try:
        for step in range(start_iter, max_iters):
            
            if main_process():
                training_stats.IterTic()

            # get the data batch
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                logger.info('Training dataloader length : ' + str(len(dataloader_iterator)))
                data = next(dataloader_iterator)
            except Exception as e:
                logger.info('dataloader errors exist : ' + str(e))
                continue
            data = to_cuda(data)
            # check training data
            # for i in range(data['target'].shape[0]):
            #     if 'DDAD' in data['dataset'][i] or \
            #         'Lyft' in data['dataset'][i] or \
            #         'DSEC' in data['dataset'][i] or \
            #         'Argovers2' in data['dataset'][i]:
            #         replace = True
            #     else:
            #         replace = False
            #     visual_train_data(data['target'][i, ...], data['input'][i,...], data['filename'][i], cfg.work_dir, replace=replace)

            # forward
            pred_depth, losses_dict, conf, output_dict = model(data)

            if 'norm_out_list' in output_dict:
                pred_norm_flag = True
                pred_norm = output_dict['norm_out_list'][-1][:, :3, :, :]
                pred_norm_kappa = output_dict['norm_out_list'][-1][:, 3:, :, :]
            else:
                pred_norm_flag = False
                pred_norm = None
                pred_norm_kappa = None

            optimizer.zero_grad()
            losses_dict['total_loss'].backward()
            optimizer.step()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(losses_dict)

            try:
                lr_scheduler.after_train_iter(optimizer)

                if main_process():
                    training_stats.update_iter_stats(loss_dict_reduced)
                    training_stats.IterToc()
                    training_stats.log_iter_stats(step, optimizer, max_iters, val_err)
                
                # save checkpoint
                if main_process():
                    if ((step+1) % save_interval == 0) or ((step+1)==max_iters):
                        save_ckpt(cfg, model, optimizer, lr_scheduler, step+1, epoch)
                        
                # validate the model
                if cfg.evaluation.online_eval and \
                    (step+1) % eval_interval == 0 and \
                    val_dataloader is not None:
                    val_err = validate(cfg, step+1, model, val_dataloader, tb_logger)
            except Exception as e:
                logger.info('error during validation : ' + str(e) + ', conitnue...')

    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)
    except Exception as e:
        logger.info('error during training : ' + str(e) + ', conitnue...')


def train_by_iters_multisize(cfg, model, optimizer, lr_scheduler, train_dataloader, val_dataloader):
    """
    Do the training by iterations.
    """
    logger = logging.getLogger()
    tb_logger = None
    if cfg.use_tensorboard and main_process():
        tb_logger = SummaryWriter(cfg.tensorboard_dir)
    if main_process():
        training_stats = TrainingStats(log_period=cfg.log_interval, tensorboard_logger=tb_logger)
    
    lr_scheduler.before_run(optimizer)
    
    # set training steps
    max_iters = cfg.runner.max_iters
    start_iter = lr_scheduler._step_count

    save_interval = cfg.checkpoint_config.interval
    eval_interval = cfg.evaluation.interval
    epoch = 0
    logger.info('Create iterator.')
    dataloader_iterator = iter(train_dataloader)

    val_err = {}
    
    logger.info('Start training.')
    try:
        for step in range(start_iter, max_iters):
            
            if main_process():
                training_stats.IterTic()

            # get the data batch
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)
            except:
                logger.info('Some training data errors exist in the current iter!')
                continue
            data = to_cuda(data)

            if step % 100 == 0:
                set_random_crop_size_for_iter(train_dataloader, step)
            
            # check training data
            # for i in range(data['target'].shape[0]):
            #     if 'DDAD' in data['dataset'][i] or \
            #         'Lyft' in data['dataset'][i] or \
            #         'DSEC' in data['dataset'][i] or \
            #         'Argovers2' in data['dataset'][i]:
            #         replace = True
            #     else:
            #         replace = False
            #     visual_train_data(data['target'][i, ...], data['input'][i,...], data['filename'][i], cfg.work_dir, replace=replace)

            # forward
            pred_depth, losses_dict, conf, output_dict = model(data)

            if 'norm_out_list' in output_dict:
                pred_norm_flag = True
                pred_norm = output_dict['norm_out_list'][-1][:, :3, :, :]
                pred_norm_kappa = output_dict['norm_out_list'][-1][:, 3:, :, :]
            else:
                pred_norm_flag = False
                pred_norm = None
                pred_norm_kappa = None

            optimizer.zero_grad()
            losses_dict['total_loss'].backward()
            optimizer.step()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(losses_dict)

            lr_scheduler.after_train_iter(optimizer)

            if main_process():
                training_stats.update_iter_stats(loss_dict_reduced)
                training_stats.IterToc()
                training_stats.log_iter_stats(step, optimizer, max_iters, val_err)

            # validate the model
            if cfg.evaluation.online_eval and \
                (step+1) % eval_interval == 0 and \
                val_dataloader is not None:
                logger.info('validate on rank: %d' %cfg.dist_params.global_rank)
                torch.distributed.barrier()
                val_err = validate(cfg, step+1, model, val_dataloader, tb_logger)

            # save checkpoint
            if main_process():
                if ((step+1) % save_interval == 0) or ((step+1)==max_iters):
                    save_ckpt(cfg, model, optimizer, lr_scheduler, step+1, epoch)

    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)


# def train_by_iters_amp(cfg, model, optimizer, lr_scheduler, train_dataloader, val_dataloader, loss_scaler):
#     """
#     Do the training by iterations.
#     Mix precision is employed.
#     """
#     # set up logger
#     tb_logger = None
#     if cfg.use_tensorboard and main_process():
#         tb_logger = SummaryWriter(cfg.tensorboard_dir)
#     logger = logging.getLogger()
#     # training status
#     if main_process():
#         training_stats = TrainingStats(log_period=cfg.log_interval, tensorboard_logger=tb_logger)

#     # learning schedule
#     lr_scheduler.before_run(optimizer)
    
#     # set training steps
#     max_iters = cfg.runner.max_iters
#     start_iter = lr_scheduler._step_count

#     save_interval = cfg.checkpoint_config.interval
#     eval_interval = cfg.evaluation.interval
#     epoch = 0
    
#     dataloader_iterator = iter(train_dataloader)

#     val_err = {}
#     # torch.cuda.empty_cache()


#     try:
#         for step in range(start_iter, max_iters):
            
#             if main_process():
#                 training_stats.IterTic()

#             # get the data batch
#             try:
#                 data = next(dataloader_iterator)
#             except:
#                 dataloader_iterator = iter(train_dataloader)
#                 data = next(dataloader_iterator)

#             data = to_cuda(data)
            
#             # # check training data
#             # for i in range(data['target'].shape[0]):
#             #     if 'DDAD' in data['dataset'][i] or \
#             #         'Lyft' in data['dataset'][i] or \
#             #         'DSEC' in data['dataset'][i] or \
#             #         'Argovers2' in data['dataset'][i]:
#             #         replace = True
#             #     else:
#             #         replace = False
#             #     visual_train_data(data['target'][i, ...], data['input'][i,...], data['filename'][i], cfg.work_dir, replace=replace)

#             # forward
#             #with torch.cuda.amp.autocast(dtype=torch.bfloat16 if is_bf16_supported() else torch.float16):
#             with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                 pred_depth, losses_dict, conf = model(data)

#             total_loss = losses_dict['total_loss']

#             if not math.isfinite(total_loss):
#                 logger.info("Loss is {}, stopping training".format(total_loss))
#                 sys.exit(1)
            
#             # optimize, backward
#             loss_scaler(total_loss, optimizer, clip_grad=10, parameters=model.parameters(), update_grad=True)

#             # reduce losses over all GPUs for logging purposes
#             loss_dict_reduced = reduce_dict(losses_dict)

#             lr_scheduler.after_train_iter(optimizer)

#             if main_process():
#                 training_stats.update_iter_stats(loss_dict_reduced)
#                 training_stats.IterToc()
#                 training_stats.log_iter_stats(step, optimizer, max_iters, val_err)

#             # validate the model
#             if cfg.evaluation.online_eval and \
#                 (step+1) % eval_interval == 0 and \
#                 val_dataloader is not None:
#                 val_err = validate(cfg, step+1, model, val_dataloader, tb_logger)

#             # save checkpoint
#             if main_process():
#                 if ((step+1) % save_interval == 0) or ((step+1)==max_iters):
#                     save_ckpt(cfg, model, optimizer, lr_scheduler, step+1, epoch, loss_scaler=loss_scaler)
            

#     except (RuntimeError, KeyboardInterrupt):
#         stack_trace = traceback.format_exc()
#         print(stack_trace)


def validate(cfg, iter, model, val_dataloader, tb_logger):
    """
    Validate the model.
    """
    model.eval()
    logger = logging.getLogger()
    # prepare dir for visualization data
    save_val_meta_data_dir = create_dir_for_validate_meta(cfg.work_dir, iter)
    save_html_path = save_val_meta_data_dir + '.html'

    # depth metric meter
    dam = MetricAverageMeter(cfg.evaluation.metrics)
    dam_global = MetricAverageMeter(cfg.evaluation.metrics)
    dam_median = MetricAverageMeter(cfg.evaluation.metrics)
    # try:
    #     normal_meter = NormalMeter(cfg.evaluation.norm_metrics)
    # except:
    #     normal_meter = NormalMeter()
    for i, data in enumerate(val_dataloader):
        if i % 100 == 0:
            logger.info('Validation step: %d' %i)
        data = to_cuda(data)
        pred_depth, confidence, output_dict = model.module.inference(data)
        pred_depth = pred_depth.squeeze()
        gt_depth = data['target'].cuda(non_blocking=True).squeeze()
        
        if 'norm_out_list' in output_dict:
            pred_norm_flag = True
            pred_norm = output_dict['norm_out_list'][-1][:, :3, :, :]
            pred_norm_kappa = output_dict['norm_out_list'][-1][:, 3:, :, :]
        else:
            pred_norm_flag = False
            pred_norm = None
            pred_norm_kappa = None
        
        pad = data['pad'].squeeze()
        H, W = pred_depth.shape
        pred_depth = pred_depth[pad[0]:H-pad[1], pad[2]:W-pad[3]]
        gt_depth = gt_depth[pad[0]:H-pad[1], pad[2]:W-pad[3]]
        rgb = data['input'][0, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        # if pred_norm_flag:
        #     pred_norm = pred_norm[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        #     pred_norm_kappa = pred_norm_kappa[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        #     gt_norm = data['norms'][0].cuda(non_blocking=True)
        #     gt_norm = gt_norm[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        #     gt_norm_mask = (gt_norm[:, 0:1, :, :] == 0) & (gt_norm[:, 1:2, :, :] == 0) & (gt_norm[:, 2:3, :, :] == 0)
        #     gt_norm_mask = ~gt_norm_mask
        #     normal_meter.update_metrics_gpu(pred_norm, gt_norm, gt_norm_mask, cfg.distributed)
        mask = gt_depth > 0
        #pred_depth_resize = cv2.resize(pred_depth.cpu().numpy(), (torch.squeeze(data['B_raw']).shape[1], torch.squeeze(data['B_raw']).shape[0]))
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, cfg.distributed)

        pred_global, _ = align_scale_shift(pred_depth, gt_depth)
        dam_global.update_metrics_gpu(pred_global, gt_depth, mask, cfg.distributed)
        
        valid_mask = (gt_depth > 0)
        pred_median = pred_depth * gt_depth[valid_mask].median() / pred_depth[valid_mask].median()
        dam_median.update_metrics_gpu(pred_median, gt_depth, mask, cfg.distributed)

        # save evaluation results
        if i % 1 == 0:
            save_val_imgs(iter, 
                        pred_depth, 
                        gt_depth, 
                        pred_norm,
                        pred_norm_kappa,
                        rgb,
                        data['filename'][0], 
                        save_val_meta_data_dir,
                        tb_logger=(tb_logger if (i==20 | i==60) else None))


    # create html for visualization
    merged_rgb_pred_gt = os.path.join(save_val_meta_data_dir, '*_merge.jpg')
    name2path = dict(merg=merged_rgb_pred_gt) #dict(rgbs=rgbs, pred=pred, gt=gt)
    if main_process():
        create_html(name2path, save_path=save_html_path, size=(256*3, 512))

    # get validation error
    eval_error = dam.get_metrics()
    globa_eval_error = dam_global.get_metrics()
    median_eval_error = dam_median.get_metrics()
    globa_eval_error_new = {'global_' + key: value for key, value in globa_eval_error.items()}
    median_eval_error_new = {'median_' + key: value for key, value in median_eval_error.items()}
    eval_error.update(globa_eval_error_new)
    eval_error.update(median_eval_error_new)

    # normal_eval_error = normal_meter.get_metrics()
    # eval_error.update(normal_eval_error)
    model.train()
    return eval_error

def set_random_crop_size_for_iter(dataloader: torch.utils.data.dataloader.DataLoader, iter: int):
    size_pool = [
        [480, 1280],
        [512, 960],
        [512, 1280],
        [576, 1024],
    ]
    random.seed(iter)
    sample = random.choice(size_pool)

    crop_size = sample
    datasets_groups = len(dataloader.dataset.datasets)
    for i in range(datasets_groups):
        for j in range(len(dataloader.dataset.datasets[i].datasets)):
            dataloader.dataset.datasets[i].datasets[j].set_random_crop_size(crop_size)
    return