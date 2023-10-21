import numpy as np
import torch
import torch.distributed as dist
from .inverse_warp import pixel2cam, cam2pixel2
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = np.longdouble(0.0)
        self.avg = np.longdouble(0.0)
        self.sum = np.longdouble(0.0)
        self.count = np.longdouble(0.0)

    def update(self, val, n: float = 1) -> None:
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / (self.count + 1e-6)

class MedianMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = np.longdouble(0.0)
        self.data = np.longdouble([0.0])
        self.median = np.longdouble(0.0)
        self.count = np.longdouble(0.0)

    def update(self, val, n: float = 1) -> None:
        self.val = val
        self.count += n
        self.data = np.concatenate([self.data, self.val], axis=0)
        self.median = np.median(self.data)

class NormalMeter(AverageMeter):
    def __init__(self, metrics=['mean', 'median', 'rmse', 'a1', 'a2', 'a3', 'a4', 'a5']) -> None:
        """ Initialize object. """
        # average meters for metrics
        self.mean = AverageMeter()
        self.median = MedianMeter()
        self.rmse = AverageMeter()
        self.a1 = AverageMeter()
        self.a2 = AverageMeter()
        self.a3 = AverageMeter()
        self.a4 = AverageMeter()
        self.a5 = AverageMeter()

        self.metrics = metrics
    
    def reset(self):
        self.mean.reset()
        self.median.reset()
        self.rmse.reset()
        self.a1.reset()
        self.a2.reset()
        self.a3.reset()
        self.a4.reset()
        self.a5.reset()
    
    def update_metrics_gpu(
        self,
        pred_norm: torch.Tensor,
        gt_norm: torch.Tensor,
        gt_norm_mask: torch.Tensor,
        is_distributed: bool):
        
        if len(pred_norm.shape) == 3:
            pred_norm = pred_norm[None, :, :, :]
            gt_norm = gt_norm[None, :, :, :]
            gt_norm_mask = gt_norm_mask[None, :, :, :]
        
        assert pred_norm.shape == gt_norm.shape

        # upsample if necessary
        if pred_norm.size(2) != gt_norm.size(2):
            pred_norm = F.interpolate(pred_norm, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
        
        prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
        prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
        E = torch.acos(prediction_error) * 180.0 / np.pi

        mask = gt_norm_mask[:, 0, :, :]
        total_normal_errors = E[mask]
        valid_pics = mask.sum()
        if is_distributed:
            dist.all_reduce(valid_pics)
        valid_pics = int(valid_pics)

        # mean
        abs_num = torch.sum(total_normal_errors)
        if is_distributed:
            dist.all_reduce(abs_num)
        abs_num = abs_num.detach().cpu().numpy()
        self.mean.update(abs_num, valid_pics)

        # rmse
        rmse_sum = torch.sum(total_normal_errors * total_normal_errors)
        if is_distributed:
            dist.all_reduce(rmse_sum)
        rmse_sum_detach = rmse_sum.detach().cpu().numpy()
        self.rmse.update(rmse_sum_detach, valid_pics)
        self.rmse.avg = np.sqrt(self.rmse.avg)

        # a1
        a1_sum = 100.0 * torch.sum(total_normal_errors < 5)
        if is_distributed:
            dist.all_reduce(a1_sum)
        a1_sum = a1_sum.detach().cpu().numpy()
        self.a1.update(a1_sum, valid_pics)

        # a2
        a2_sum = 100.0 * torch.sum(total_normal_errors < 7.5)
        if is_distributed:
            dist.all_reduce(a2_sum)
        a2_sum = a2_sum.detach().cpu().numpy()
        self.a2.update(a2_sum, valid_pics)

        # a3
        a3_sum = 100.0 * torch.sum(total_normal_errors < 11.25)
        if is_distributed:
            dist.all_reduce(a3_sum)
        a3_sum = a3_sum.detach().cpu().numpy()
        self.a3.update(a3_sum, valid_pics)

        # a4
        a4_sum = 100.0 * torch.sum(total_normal_errors < 22.5)
        if is_distributed:
            dist.all_reduce(a4_sum)
        a4_sum = a4_sum.detach().cpu().numpy()
        self.a4.update(a4_sum, valid_pics)

        # a5
        a5_sum = 100.0 * torch.sum(total_normal_errors < 30)
        if is_distributed:
            dist.all_reduce(a5_sum)
        a5_sum = a5_sum.detach().cpu().numpy()
        self.a5.update(a5_sum, valid_pics)

    def get_metrics(self):
        metrics_dict = {}
        for metric in self.metrics:
            if metric == 'median':
                metrics_dict[metric] = round(self.__getattribute__(metric).median, 3)
            else:
                metrics_dict[metric] = round(self.__getattribute__(metric).avg)
        return metrics_dict

class MetricAverageMeter(AverageMeter):
    """ 
    An AverageMeter designed specifically for evaluating segmentation results.
    """
    def __init__(self, metrics: list) -> None:
        """ Initialize object. """
        # average meters for metrics
        self.abs_rel = AverageMeter()
        self.rmse = AverageMeter()
        self.silog = AverageMeter()
        self.delta1 = AverageMeter()
        self.delta2 = AverageMeter()
        self.delta3 = AverageMeter()

        self.metrics = metrics

        self.consistency = AverageMeter()
        self.log10 = AverageMeter()
        self.rmse_log = AverageMeter()
        self.sq_rel = AverageMeter()
    
    def reset(self):
        self.abs_rel.reset()
        self.rmse.reset()
        self.silog.reset()
        self.delta1.reset()
        self.delta2.reset()
        self.delta3.reset()
        self.consistency.reset()
        self.log10.reset()
        self.rmse_log.reset()
        self.sq_rel.reset()

    def update_metrics_cpu(self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,):
        '''
        Update metrics on cpu
        '''

        assert pred.shape == target.shape

        if len(pred.shape) == 3:
            pred = pred[:, None, :, :]
            target = target[:, None, :, :]
            mask = mask[:, None, :, :]
        elif len(pred.shape) == 2:
            pred = pred[None, None, :, :]
            target = target[None, None, :, :]
            mask = mask[None, None, :, :]
        

        # Absolute relative error
        abs_rel_sum, valid_pics = get_absrel_err(pred, target, mask)
        abs_rel_sum = abs_rel_sum.numpy()
        valid_pics = valid_pics.numpy()
        self.abs_rel.update(abs_rel_sum, valid_pics)

        # squraed relative error
        sqrel_sum, _ = get_sqrel_err(pred, target, mask)
        sqrel_sum = sqrel_sum.numpy()
        self.sq_rel.update(sqrel_sum, valid_pics)

        # root mean squared error
        rmse_sum, _ = get_rmse_err(pred, target, mask)
        rmse_sum = rmse_sum.numpy()
        self.rmse.update(rmse_sum, valid_pics)

        # log root mean squared error
        log_rmse_sum, _ = get_rmse_log_err(pred, target, mask)
        log_rmse_sum = log_rmse_sum.numpy()
        self.rmse_log.update(log_rmse_sum, valid_pics)

        # log10 error
        log10_sum, _ = get_log10_err(pred, target, mask)
        log10_sum = log10_sum.numpy()
        self.log10.update(log10_sum, valid_pics)

        # scale-invariant root mean squared error in log space
        silog_sum, _ = get_silog_err(pred, target, mask)
        silog_sum = silog_sum.numpy()
        self.silog.update(silog_sum, valid_pics)

        # ratio error, delta1, ...
        delta1_sum, delta2_sum, delta3_sum, _ = get_ratio_err(pred, target, mask)
        delta1_sum = delta1_sum.numpy()
        delta2_sum = delta2_sum.numpy()
        delta3_sum = delta3_sum.numpy()

        self.delta1.update(delta1_sum, valid_pics)
        self.delta2.update(delta2_sum, valid_pics)
        self.delta3.update(delta3_sum, valid_pics)


    def update_metrics_gpu(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        is_distributed: bool,
        pred_next: torch.Tensor = None,
        pose_f1_to_f2: torch.Tensor = None,
        intrinsic: torch.Tensor = None,
        ):
        '''
        Update metric on GPU. It supports distributed processing. If multiple machines are employed, please
        set 'is_distributed' as True.
        '''
        assert pred.shape == target.shape

        if len(pred.shape) == 3:
            pred = pred[:, None, :, :]
            target = target[:, None, :, :]
            mask = mask[:, None, :, :]
        elif len(pred.shape) == 2:
            pred = pred[None, None, :, :]
            target = target[None, None, :, :]
            mask = mask[None, None, :, :]
        

        # Absolute relative error
        abs_rel_sum, valid_pics = get_absrel_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(abs_rel_sum), dist.all_reduce(valid_pics)
        abs_rel_sum = abs_rel_sum.cpu().numpy()
        valid_pics = int(valid_pics)
        self.abs_rel.update(abs_rel_sum, valid_pics)

        # squraed relative error
        sqrel_sum, _ = get_sqrel_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(sqrel_sum)
        sqrel_sum = sqrel_sum.cpu().numpy()
        self.sq_rel.update(sqrel_sum, valid_pics)

        # root mean squared error
        rmse_sum, _ = get_rmse_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(rmse_sum)
        rmse_sum = rmse_sum.cpu().numpy()
        self.rmse.update(rmse_sum, valid_pics)

        # log root mean squared error
        log_rmse_sum, _ = get_rmse_log_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(log_rmse_sum)
        log_rmse_sum = log_rmse_sum.cpu().numpy()
        self.rmse_log.update(log_rmse_sum, valid_pics)

        # log10 error
        log10_sum, _ = get_log10_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(log10_sum)
        log10_sum = log10_sum.cpu().numpy()
        self.log10.update(log10_sum, valid_pics)

        # scale-invariant root mean squared error in log space
        silog_sum, _ = get_silog_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(silog_sum)
        silog_sum = silog_sum.cpu().numpy()
        self.silog.update(silog_sum, valid_pics)

        # ratio error, delta1, ...
        delta1_sum, delta2_sum, delta3_sum, _ = get_ratio_err(pred, target, mask)
        if is_distributed:
            dist.all_reduce(delta1_sum), dist.all_reduce(delta2_sum), dist.all_reduce(delta3_sum)
        delta1_sum = delta1_sum.cpu().numpy()
        delta2_sum = delta2_sum.cpu().numpy()
        delta3_sum = delta3_sum.cpu().numpy()

        self.delta1.update(delta1_sum, valid_pics)
        self.delta2.update(delta2_sum, valid_pics)
        self.delta3.update(delta3_sum, valid_pics)

        # video consistency error
        consistency_rel_sum, valid_warps = get_video_consistency_err(pred, pred_next, pose_f1_to_f2, intrinsic)
        if is_distributed:
            dist.all_reduce(consistency_rel_sum), dist.all_reduce(valid_warps)
        consistency_rel_sum = consistency_rel_sum.cpu().numpy()
        valid_warps = int(valid_warps)
        self.consistency.update(consistency_rel_sum, valid_warps)


    def get_metrics(self):
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric] = round(self.__getattribute__(metric).avg, 3)
        return metrics_dict

def get_absrel_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes absolute relative error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    # Mean Absolute Relative Error
    rel = torch.abs(t_m - p_m) / (t_m + 1e-10) # compute errors
    abs_rel_sum = torch.sum(rel.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    abs_err = abs_rel_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(abs_err), valid_pics

def get_sqrel_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes squared relative error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    # squared Relative Error
    sq_rel = torch.abs(t_m - p_m) ** 2 / (t_m + 1e-10) # compute errors
    sq_rel_sum = torch.sum(sq_rel.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    sqrel_err = sq_rel_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(sqrel_err), valid_pics

def get_log10_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log10 error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    log10_diff = torch.abs(diff_log)
    log10_sum = torch.sum(log10_diff.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    log10_err = log10_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(log10_err), valid_pics

def get_rmse_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    square = (t_m - p_m) ** 2
    rmse_sum = torch.sum(square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    rmse = torch.sqrt(rmse_sum / (num + 1e-10))
    valid_pics = torch.sum(num > 0)
    return torch.sum(rmse), valid_pics

def get_rmse_log_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    square = diff_log ** 2
    rmse_log_sum = torch.sum(square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    rmse_log = torch.sqrt(rmse_log_sum / (num + 1e-10))
    valid_pics = torch.sum(num > 0)
    return torch.sum(rmse_log), valid_pics

def get_silog_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    diff_log_sum = torch.sum(diff_log.reshape((b, c, -1)), dim=2) # [b, c]
    diff_log_square = diff_log ** 2
    diff_log_square_sum = torch.sum(diff_log_square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    silog = torch.sqrt(diff_log_square_sum / (num + 1e-10) - (diff_log_sum / (num + 1e-10)) ** 2)
    valid_pics = torch.sum(num > 0)
    return torch.sum(silog), valid_pics

def get_ratio_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """
    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)
    pred_gt = p_m / (t_m + 1e-10)
    gt_pred = gt_pred.reshape((b, c, -1))
    pred_gt = pred_gt.reshape((b, c, -1))
    gt_pred_gt = torch.cat((gt_pred, pred_gt), axis=1)
    ratio_max = torch.amax(gt_pred_gt, axis=1)

    delta_1_sum = torch.sum((ratio_max < 1.25), dim=1) # [b, ]
    delta_2_sum = torch.sum((ratio_max < 1.25 ** 2), dim=1) # [b, ]
    delta_3_sum = torch.sum((ratio_max < 1.25 ** 3), dim=1) # [b, ]
    num = torch.sum(mask.reshape((b, -1)), dim=1) # [b, ]

    delta_1 = delta_1_sum / (num + 1e-10)
    delta_2 = delta_2_sum / (num + 1e-10)
    delta_3 = delta_3_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)    

    return torch.sum(delta_1), torch.sum(delta_2), torch.sum(delta_3), valid_pics

def unproj_pcd(
    depth: torch.tensor,
    intrinsic: torch.tensor,
):
    depth = depth.squeeze(1) # [B, H, W]
    b, h, w = depth.size()
    v = torch.arange(0, h).view(1, h, 1).expand(b, h, w).type_as(depth) # [B, H, W]
    u = torch.arange(0, w).view(1, 1, w).expand(b, h, w).type_as(depth) # [B, H, W]
    x = (u - intrinsic[:, 0, 2]) / intrinsic[:, 0, 0] * depth # [B, H, W]
    y = (v - intrinsic[:, 1, 2]) / intrinsic[:, 0, 0] * depth # [B, H, W]
    pcd = torch.stack([x, y, depth], dim=1)
    return pcd

def forward_warp(
    depth: torch.tensor,
    intrinsic: torch.tensor,
    pose: torch.tensor,
):
    """
    Warp the depth with provided pose.
    Args:
        depth: depth map of the target image -- [B, 1, H, W]
        intrinsic: camera intrinsic parameters -- [B, 3, 3]
        pose: the camera pose -- [B, 4, 4]
    """
    B, _, H, W = depth.shape
    pcd = unproj_pcd(depth.float(), intrinsic.float())
    pcd = pcd.reshape(B, 3, -1) # [B, 3, H*W]
    rot, tr = pose[:, :3, :3], pose[:, :3, -1:]
    proj_pcd = rot @ pcd + tr

    img_coors = intrinsic @ proj_pcd

    X = img_coors[:, 0, :]
    Y = img_coors[:, 1, :]
    Z = img_coors[:, 2, :].clamp(min=1e-3)

    x_img_coor = (X/Z + 0.5).long()
    y_img_coor = (Y/Z + 0.5).long()
    
    X_mask = ((x_img_coor >= 0) & (x_img_coor < W))
    Y_mask = ((y_img_coor >= 0) & (y_img_coor < H))    
    mask = X_mask & Y_mask

    proj_depth = torch.zeros_like(Z).reshape(B, 1, H, W)
    for i in range(B):
        proj_depth[i, :, y_img_coor[i, ...][mask[i, ...]], x_img_coor[i, ...][mask[i, ...]]] = Z[i, ...][mask[i, ...]]
    plt.imsave('warp2.png', proj_depth.squeeze().cpu().numpy(), cmap='rainbow')
    return proj_depth


def get_video_consistency_err(
    pred_f1: torch.tensor,
    pred_f2: torch.tensor,
    ego_pose_f1_to_f2: torch.tensor,
    intrinsic: torch.tensor,
):
    """
    Compute consistency error between consecutive frames.
    """
    if pred_f2 is None or ego_pose_f1_to_f2 is None or intrinsic is None:
        return torch.zeros_like(pred_f1).sum(), torch.zeros_like(pred_f1).sum()
    ego_pose_f1_to_f2 = ego_pose_f1_to_f2.float()
    pred_f2 = pred_f2.float()

    pred_f1 = pred_f1[:, None, :, :] if pred_f1.ndim == 3 else pred_f1
    pred_f2 = pred_f2[:, None, :, :] if pred_f2.ndim == 3 else pred_f2
    pred_f1 = pred_f1[None, None, :, :] if pred_f1.ndim == 2 else pred_f1
    pred_f2 = pred_f2[None, None, :, :] if pred_f2.ndim == 2 else pred_f2

    B, _, H, W = pred_f1.shape
    cam_coords = pixel2cam(pred_f1.squeeze(1).float(), intrinsic.inverse().float())  # [B,3,H,W]
    
    proj_f1_to_f2 = intrinsic @ ego_pose_f1_to_f2[:, :3, :] # [B, 3, 4]
    rot, tr = proj_f1_to_f2[:, :, :3], proj_f1_to_f2[:, :, -1:]
    f2_pixel_coords, warped_depth_f1_to_f2 = cam2pixel2(cam_coords, rot, tr, padding_mode='zeros')  # [B,H,W,2]
    
    projected_depth = F.grid_sample(pred_f2, f2_pixel_coords, padding_mode="zeros", align_corners=False)

    mask_valid = (projected_depth > 1e-6) & (warped_depth_f1_to_f2 > 1e-6)

    consistency_rel_err, valid_pix = get_absrel_err(warped_depth_f1_to_f2, projected_depth, mask_valid)
    return consistency_rel_err, valid_pix

if __name__ == '__main__':
    cfg = ['abs_rel', 'delta1']
    dam = MetricAverageMeter(cfg)

    pred_depth = np.random.random([2, 480, 640])
    gt_depth = np.random.random([2, 480, 640]) - 0.5
    intrinsic = [[100, 100, 200, 200], [200, 200, 300, 300]]
    
    pred = torch.from_numpy(pred_depth).cuda()
    gt = torch.from_numpy(gt_depth).cuda()

    mask = gt > 0
    dam.update_metrics_gpu(pred, gt, mask, False)
    eval_error = dam.get_metrics()
    print(eval_error)
    