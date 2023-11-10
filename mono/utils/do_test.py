import torch
import torch.nn.functional as F
import logging
import os
import os.path as osp
from mono.utils.avg_meter import MetricAverageMeter, NormalMeter
from mono.utils.visualization import save_val_imgs, visual_train_data, create_html, save_raw_imgs
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud
from mono.utils.transform import gray_to_colormap

from mono.utils.lwlr import sparse_depth_lwlr_batch

# from scipy.sparse import csr_array
from scipy import sparse

img_file_type = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
np_file_type = ['.npz', '.npy']
JPG_file_type = ['.JPG']

def load_data(path: str, is_rgb_img: bool=False):
    if not osp.exists(path):
        raise RuntimeError(f'{path} does not exist.')

    data_type = osp.splitext(path)[-1]
    if data_type in img_file_type:
        if is_rgb_img:
            data = cv2.imread(path)
        else:
            data = cv2.imread(path, -1)
    elif data_type in np_file_type:
        data = np.load(path)
    elif data_type in JPG_file_type:
        # NOTE: only support .JPG file depth of ETH3D so far.
        if is_rgb_img:
            data = cv2.imread(path)
        else:
            f = open(path, 'r')
            data = np.fromfile(f, np.float32)
            data = data.reshape((4032, 6048))
    else:
        raise RuntimeError(f'{data_type} is not supported in current version.')
    
    return data.squeeze()

def to_cuda(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>=1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scaled = pred * scale
    return pred_scaled, scale

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

def align_scale_shift_numpy(pred: np.array, target: np.array):
    mask = target > 0
    target_mask = target[mask]
    pred_mask = pred[mask]
    if np.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale


def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map. 
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model

def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio, model_type):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    if 'convnext' in model_type:
        padding = [123.675, 116.28, 103.53]
    elif ('beit' in model_type) or ('swinv2' in model_type):
        padding = [127.5, 127.5, 127.5]
    else:
        raise ValueError

    
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding)
    
    # Resize, adjust principle point
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=-1)

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor=1/to_scale_ratio
    return image, cam_model, pad, label_scale_factor


def transform_test_data(rgb, intrinsic, data_basic, model_type):
    """
    Pre-process the input for fowarding.
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter. [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic['canonocal_space']
    forward_size = data_basic.crop_size
    if 'convnext' in model_type:
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    elif ('beit' in model_type) or ('swinv2' in model_type):
        mean = torch.tensor([127.5, 127.5, 127.5]).float()[:, None, None]
        std = torch.tensor([127.5, 127.5, 127.5]).float()[:, None, None]
    else:
        raise ValueError

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0
    to_canonical_ratio = canonical_space['focal_length'] / ori_focal
    canonical_intrinsic = [
        canonical_space['focal_length'],
        canonical_space['focal_length'],
        intrinsic[2]*to_canonical_ratio,
        intrinsic[3]*to_canonical_ratio,
    ]
        
    h_canonical = int(ori_h * to_canonical_ratio + 0.5)
    w_canonical = int(ori_w * to_canonical_ratio + 0.5)

    # resize 
    rgb, cam_model, pad, label_scale_factor = resize_for_input(rgb, forward_size, canonical_intrinsic, [h_canonical, w_canonical], to_canonical_ratio)

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda()
    cam_models_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_models_stacks, pad, label_scale_factor


def get_prediction(
    model: torch.nn.Module,
    input: torch.tensor,
    cam_model: torch.tensor,
    pad_info: torch.tensor,
    scale_info: torch.tensor,
    gt_depth: torch.tensor,
    gt_norm: torch.tensor,
    normalize_scale: float,
    ori_shape: list=[],
):

    data = dict(
        input=input,
        cam_model=cam_model,
        norms=[gt_norm],
    )
    pred_depth, confidence, output_dict = model.module.inference(data)
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    if gt_depth is not None:
        resize_shape = gt_depth.shape
    elif ori_shape != []:
        resize_shape = ori_shape
    else:
        resize_shape = pred_depth.shape

    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], resize_shape, mode='bilinear').squeeze() # to original size
    pred_depth = pred_depth * normalize_scale / scale_info
    if gt_depth is not None:
        pred_depth_scale, scale = align_scale(pred_depth, gt_depth)
    else:
        pred_depth_scale = None
        scale = None
    
    # norms
    if 'norm_out_list' in output_dict:
        pred_norm = output_dict['norm_out_list'][-1][:, :3, :, :]
        pred_norm_kappa = output_dict['norm_out_list'][-1][:, 3:, :, :]

        pred_norm = pred_norm[:, :, pad_info[0] : pred_norm.shape[2] - pad_info[1], pad_info[2] : pred_norm.shape[3] - pad_info[3]]
        pred_norm_kappa = pred_norm_kappa[:, :, pad_info[0] : pred_norm_kappa.shape[2] - pad_info[1], pad_info[2] : pred_norm_kappa.shape[3] - pad_info[3]]
        pred_norm = torch.nn.functional.interpolate(pred_norm, resize_shape, mode='bilinear')
        pred_norm_kappa = torch.nn.functional.interpolate(pred_norm_kappa, resize_shape, mode='bilinear')
    else:
        pred_norm = None
        pred_norm_kappa = None

    return pred_depth, pred_depth_scale, scale, pred_norm, pred_norm_kappa

def transform_test_data_scalecano(rgb, intrinsic, data_basic, model_type):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    if 'convnext' in model_type:
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    elif ('beit' in model_type) or ('swinv2' in model_type):
        mean = torch.tensor([127.5, 127.5, 127.5]).float()[:, None, None]
        std = torch.tensor([127.5, 127.5, 127.5]).float()[:, None, None]
    else:
        raise ValueError

    # BGR to RGB
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']

    cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0, model_type)

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda()
    cam_model_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor

def do_scalecano_test_with_custom_data(
    model: torch.nn.Module,
    cfg: dict,
    test_data: list,
    logger: logging.RootLogger,
    is_distributed: bool = True,
    local_rank: int = 0,
):

    show_dir = cfg.show_dir
    save_interval = 1
    save_imgs_dir = show_dir + '/vis'
    os.makedirs(save_imgs_dir, exist_ok=True)
    save_pcd_dir = show_dir + '/pcd'
    os.makedirs(save_pcd_dir, exist_ok=True)
    save_gt_pcd_dir = show_dir + '/gt_pcd'
    os.makedirs(save_gt_pcd_dir, exist_ok=True)

    normalize_scale = cfg.data_basic.depth_range[1]
    cfg.test_metrics = ['abs_rel', 'rmse', 'delta1']
    dam = MetricAverageMeter(cfg.test_metrics)
    dam_median = MetricAverageMeter(cfg.test_metrics)
    dam_global = MetricAverageMeter(cfg.test_metrics)
    # dam_lwlr = MetricAverageMeter(cfg.test_metrics)

    model_type = cfg.model.backbone.type
    try:
        normal_meter = NormalMeter(cfg.evaluation.norm_metrics)
    except:
        normal_meter = NormalMeter()
    
    for i, an in tqdm(enumerate(test_data)):
        rgb_origin = cv2.imread(an['rgb'])[:, :, ::-1].copy()
        print('rgb_path :', an['rgb'])
        if an['depth'] is not None:
            gt_depth = load_data(an['depth'])
            # NOTE: 0 for invalid mask of gt_depth_mask
            if ('depth_mask' in an) and (an['depth_mask'] is not None):
                gt_depth_mask = load_data(an['depth_mask'])
                gt_depth_mask = cv2.resize(gt_depth_mask, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_depth[gt_depth_mask == 0] = 0

            # if an['depth'].endswith('.npy'):
            #     gt_depth = np.load(an['depth'])
            # elif an['depth'].endswith('.npz'):
            #     gt_depth = sparse.load_npz(an['depth']).toarray()
            # else:
            #     gt_depth = cv2.imread(an['depth'], -1)

            gt_depth_scale = an['depth_scale']
            gt_depth = gt_depth / gt_depth_scale
            gt_depth[gt_depth > 65535] = 0 # a causial gt_depth max
            gt_depth_flag = True
        else:
            gt_depth = None
            gt_depth_flag = False
        intrinsic = an['intrinsic']
        if intrinsic is None:
            intrinsic = [1000.0, 1000.0, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]
        if an['norm'] is not None:
            gt_norm = Image.open(an['norm']).convert("RGB")
            gt_norm = np.array(gt_norm).astype(np.uint8)
            gt_norm = ((gt_norm.astype(np.float32) / 255.0) * 2.0) - 1.0
            gt_norm_flag = True
        else:
            gt_norm = None
            gt_norm_flag = False
        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic, model_type)

        pred_depth, pred_depth_scale, scale, pred_norm, pred_norm_kappa = get_prediction(
            model = model,
            input = rgb_input,
            cam_model = cam_models_stacks,
            pad_info = pad,
            scale_info = label_scale_factor,
            gt_depth = None,
            gt_norm = None,
            normalize_scale = normalize_scale,
            ori_shape=[rgb_origin.shape[0], rgb_origin.shape[1]],
        )

        if gt_depth_flag:

            pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear').squeeze() # to original size
            rgb_resize_ratio_h = gt_depth.shape[0] / rgb_origin.shape[0]; rgb_resize_ratio_w = gt_depth.shape[1] / rgb_origin.shape[1]
            intrinsic[0] = intrinsic[0] * rgb_resize_ratio_w
            intrinsic[1] = intrinsic[1] * rgb_resize_ratio_h
            intrinsic[2] = intrinsic[2] * rgb_resize_ratio_w
            intrinsic[3] = intrinsic[3] * rgb_resize_ratio_h
            rgb_origin = cv2.resize(rgb_origin, (gt_depth.shape[1], gt_depth.shape[0]), cv2.INTER_LINEAR)
            
            gt_depth = torch.from_numpy(gt_depth).cuda()

            pred_depth_median = pred_depth * gt_depth[gt_depth > 1e-8].median() / pred_depth[gt_depth > 1e-8].median()
            print('pred_depth_median :', gt_depth[gt_depth > 1e-8].median() / pred_depth[gt_depth > 1e-8].median())
            pred_global, _ = align_scale_shift(pred_depth, gt_depth)
            # pred_lwlr = torch.from_numpy(pred_lwlr).cuda()
            
            mask = (gt_depth > 1e-8)
            dam.update_metrics_gpu(pred_depth, gt_depth, mask, is_distributed)
            dam_median.update_metrics_gpu(pred_depth_median, gt_depth, mask, is_distributed)
            dam_global.update_metrics_gpu(pred_global, gt_depth, mask, is_distributed)
            # dam_lwlr.update_metrics_gpu(pred_lwlr, gt_depth, mask, is_distributed)

        if pred_norm is not None and gt_norm_flag == True:
            gt_norm = torch.from_numpy(gt_norm)[None, ...].cuda()
            gt_norm_mask = (gt_norm[:, 0:1, :, :] == 0) & (gt_norm[:, 1:2, :, :] == 0) & (gt_norm[:, 2:3, :, :] == 0)
            gt_norm_mask = ~gt_norm_mask
            normal_meter.update_metrics_gpu(pred_norm, gt_norm, gt_norm_mask, is_distributed)
        
        if i % save_interval == 0:
            os.makedirs(osp.join(save_imgs_dir, an['folder']), exist_ok=True)
            rgb_torch = torch.from_numpy(rgb_origin).to(pred_depth.device).permute(2, 0, 1)

            if 'convnext' in model_type:
                mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(rgb_torch.device)
                std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(rgb_torch.device)
            elif ('beit' in model_type) or ('swinv2' in model_type):
                mean = torch.tensor([127.5, 127.5, 127.5]).float()[:, None, None].to(rgb_torch.device)
                std = torch.tensor([127.5, 127.5, 127.5]).float()[:, None, None].to(rgb_torch.device)
            else:
                raise ValueError

            rgb_torch = torch.div((rgb_torch - mean), std)

            save_val_imgs(
                i,
                pred_depth,
                gt_depth if gt_depth is not None else torch.ones_like(pred_depth, device=pred_depth.device),
                pred_norm,
                pred_norm_kappa,
                rgb_torch,
                osp.join(an['folder'], an['filename']),
                save_imgs_dir,
                model_type,
            )

            # pcd
            pred_depth = pred_depth.detach().cpu().numpy()
            # pred_depth_pcd = cv2.resize(pred_depth, (rgb_origin.shape[1], rgb_origin.shape[0]), cv2.INTER_LINEAR)
            pcd = reconstruct_pcd(pred_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
            os.makedirs(osp.join(save_pcd_dir, an['folder']), exist_ok=True)
            save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_pcd_dir, an['folder'], an['filename'][:-4]+'.ply'))

            if gt_depth_flag:
                # gt_pcd
                gt_depth = gt_depth.detach().cpu().numpy()
                pcd = reconstruct_pcd(gt_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
                os.makedirs(osp.join(save_gt_pcd_dir, an['folder']), exist_ok=True)
                save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_gt_pcd_dir, an['folder'], an['filename'][:-4]+'.ply'))
            
            # npy
            np.save(osp.join(save_imgs_dir, an['folder'], an['filename'][:-4]+'.npy'), pred_depth)
            
            # rgb, gt_depth, and pred_depth
            cv2.imwrite(osp.join(save_imgs_dir, an['folder'], an['filename'][:-4] + '_rgb.png'), rgb_origin[:, :, ::-1])
            if gt_depth_flag:
                plt.imsave(osp.join(save_imgs_dir, an['folder'], an['filename'][:-4] + '_gt_depth.png'), gray_to_colormap(gt_depth))
            plt.imsave(osp.join(save_imgs_dir, an['folder'], an['filename'][:-4] + '_pred_depth.png'), gray_to_colormap(pred_depth))

            if gt_depth_flag:
                eval_error = dam.get_metrics()
                print('w/o match :', eval_error)

                eval_error_median = dam_median.get_metrics()
                print('median match :', eval_error_median)

                eval_error_global = dam_global.get_metrics()
                print('global match :', eval_error_global)

                # eval_error_lwlr = dam_lwlr.get_metrics()
                # print('lwlr match :', eval_error_lwlr)

            else:
                print('missing gt_depth, only save visualizations...')
    
    logger.info('Evaluation finished.')


def do_scalecano_test_with_dataloader(
    model: torch.nn.Module,
    cfg: dict,
    test_dataloder: torch.utils.data.DataLoader,
    logger: logging.RootLogger,
    is_distributed: bool = True,
    local_rank: int = 0,
):

    show_dir = cfg.show_dir
    save_interval = 1
    save_imgs_dir = show_dir + '/vis'
    os.makedirs(save_imgs_dir, exist_ok=True)
    save_pcd_dir = show_dir + '/pcd'
    os.makedirs(save_pcd_dir, exist_ok=True)

    save_gt_pcd_dir = show_dir + '/gt_pcd'
    os.makedirs(save_gt_pcd_dir, exist_ok=True)

    cfg.test_metrics = ['abs_rel', 'rmse', 'delta1']

    normalize_scale = cfg.data_basic.depth_range[1]
    dam = MetricAverageMeter(cfg.test_metrics)
    dam_median = MetricAverageMeter(cfg.test_metrics)
    dam_global = MetricAverageMeter(cfg.test_metrics)
    dam_lwlr = MetricAverageMeter(cfg.test_metrics)

    model_type = cfg.model.backbone.type

    try:
        normal_meter = NormalMeter(cfg.evaluation.norm_metrics)
    except:
        normal_meter = NormalMeter()
    
    for i, data in tqdm(enumerate(test_dataloder)):

        data = to_cuda(data)
        pred_depth, confidence, output_dict = model.module.inference(data)
        pred_depth = pred_depth.squeeze()
        # gt_depth = data['target'].cuda(non_blocking=True).squeeze()
        gt_depth = data['raw_depth'].cuda(non_blocking=True).squeeze()
        pred_depth = pred_depth / data['scale']
        
        if 'norm_out_list' in output_dict:
            pred_norm_flag = True
            pred_norm = output_dict['norm_out_list'][-1][:, :3, :, :]
            pred_norm_kappa = output_dict['norm_out_list'][-1][:, 3:, :, :]
        else:
            pred_norm_flag = False
            pred_norm = None
            pred_norm_kappa = None
        
        pad = data['pad']
        H, W = pred_depth.shape
        pred_depth = pred_depth[pad[0]:H-pad[1], pad[2]:W-pad[3]]
        # gt_depth = gt_depth[pad[0]:H-pad[1], pad[2]:W-pad[3]]
        rgb = data['input'][0, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        pred_depth = F.interpolate(pred_depth[None, None], (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear')[0, 0]
        rgb = F.interpolate(rgb[None], (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear')[0]

        if pred_norm_flag:
            pred_norm = pred_norm[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            pred_norm_kappa = pred_norm_kappa[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            pred_norm = F.interpolate(pred_norm, (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear')
            pred_norm_kappa = F.interpolate(pred_norm_kappa, (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear')
            gt_norm = data['norms'][0].cuda(non_blocking=True)
            gt_norm = gt_norm[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
            gt_norm_mask = (gt_norm[:, 0:1, :, :] == 0) & (gt_norm[:, 1:2, :, :] == 0) & (gt_norm[:, 2:3, :, :] == 0)
            gt_norm_mask = ~gt_norm_mask
            normal_meter.update_metrics_gpu(pred_norm, gt_norm, gt_norm_mask, cfg.distributed)

        mask = gt_depth > 0
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, is_distributed)
        
        pred_global, _ = align_scale_shift(pred_depth, gt_depth)
        dam_global.update_metrics_gpu(pred_global, gt_depth, mask, is_distributed)

        global_negative_percent = (pred_global < 0).sum() / (pred_global == pred_global).sum()
        logger.info('global_negative_percent :' + str(global_negative_percent))

        valid_mask = (gt_depth > 0)
        if 'depth_path' in data.keys():
            logger.info('depth_path :' + str(data['depth_path']))
        median_scale = gt_depth[valid_mask].median() / pred_depth[valid_mask].median()
        logger.info('median_scale : ' + str(median_scale))
        # logger.info('median depth value: ' + str(gt_depth[valid_mask].median()))
        pred_median = pred_depth * median_scale
        dam_median.update_metrics_gpu(pred_median, gt_depth, mask, is_distributed)

        # pred_lwlr = sparse_depth_lwlr_batch(pred_global[None], gt_depth[None], device=pred_global.device).squeeze()
        # dam_lwlr.update_metrics_gpu(pred_lwlr, gt_depth, mask, is_distributed) 
        
        filepath = data['filepath'][0]
        filename = filepath.replace('/', '_')
        intrinsic = [data['raw_intrinsic'][i].detach().cpu().numpy() for i in range(4)]
        if i % 1 == 0:
            raw_rgb = data['raw_rgb'][0].squeeze().detach().cpu().numpy()
            ratio_h = gt_depth.shape[0] / raw_rgb.shape[0]; ratio_w = gt_depth.shape[1] / raw_rgb.shape[1]
            raw_rgb = cv2.resize(raw_rgb, (gt_depth.shape[1], gt_depth.shape[0]), cv2.INTER_LINEAR)[:, :, ::-1]
            save_val_imgs(
                i,
                pred_depth,
                gt_depth if gt_depth is not None else torch.ones_like(pred_depth, device=pred_depth.device),
                pred_norm,
                pred_norm_kappa,
                rgb,
                filename,
                save_imgs_dir,
                model_type,
            )

            # pcd
            pred_depth = pred_depth.detach().cpu().numpy().astype(np.float32)
            pcd = reconstruct_pcd(pred_depth.astype(np.float32), intrinsic[0] * ratio_w, intrinsic[1] * ratio_h, intrinsic[2] * ratio_w, intrinsic[3] * ratio_h)
            save_point_cloud(pcd.reshape((-1, 3)), raw_rgb.reshape(-1, 3), osp.join(save_pcd_dir, filename[:-4]+'.ply'))

            # gt_pcd
            gt_depth = gt_depth.detach().cpu().numpy()
            pcd = reconstruct_pcd(gt_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
            save_point_cloud(pcd.reshape((-1, 3)), raw_rgb.reshape(-1, 3), osp.join(save_gt_pcd_dir, filename[:-4]+'.ply'))

            eval_error = dam.get_metrics()
            logger.info('w/o match :' + str(eval_error))
            eval_error_median = dam_median.get_metrics()
            logger.info('median match :' + str(eval_error_median))
            eval_error_global = dam_global.get_metrics()
            logger.info('global match :' + str(eval_error_global))
            # eval_error_lwlr = dam_lwlr.get_metrics()
            # print('lwlr match :', eval_error_lwlr)
    
    logger.info('Final evaluation results:')
    eval_error = dam.get_metrics()
    logger.info('w/o match :' + str(eval_error))

    eval_error_median = dam_median.get_metrics()
    logger.info('median match :' + str(eval_error_median))

    eval_error_global = dam_global.get_metrics()
    logger.info('global match :' + str(eval_error_global))

    # eval_error_lwlr = dam_lwlr.get_metrics()
    # print('lwlr match :', eval_error_lwlr)

    logger.info('Evaluation finished.')


# Generate a double-input depth estimation
def global_merge(pix2pixmodel, low_res, high_res, pix2pixsize):
    # Generate the low resolution estimation
    estimate1 = low_res
    # Resize to the inference size of merge network
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
    depth_min = estimate1.min()
    depth_max = estimate1.max()

    if depth_max - depth_min > np.finfo("float").eps:
        estimate1 = (estimate1 - depth_min) / (depth_max - depth_min)
    else:
        estimate1 = 0
    

    # Generate the high resolution estimation
    estimate2 = high_res
    # Resize to the inference size of merge network
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
    depth_min = estimate2.min()
    depth_max = estimate2.max()

    if depth_max - depth_min > np.finfo("float").eps:
        estimate2 = (estimate2 - depth_min) / (depth_max - depth_min)
    else:
        estimate2 = 0
    
    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped