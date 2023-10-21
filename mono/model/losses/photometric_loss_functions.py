import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# from .inverse_warp import inverse_warp
# from .mask_ranking_loss import Mask_Ranking_Loss
# from .normal_ranking_loss import EdgeguidedNormalRankingLoss

from mono.utils.inverse_warp import inverse_warp2

# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# compute_ssim_loss = SSIM().to(device)

# normal_ranking_loss = EdgeguidedNormalRankingLoss().to(device)

# mask_ranking_loss = Mask_Ranking_Loss().to(device)

class PhotometricGeometricLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(PhotometricGeometricLoss, self).__init__()
        self.no_min_optimize = False
        self.no_auto_mask = False
        self.return_dynamic_mask = True
        self.ssim_loss = SSIM()
        self.no_ssim = False
        self.no_dynamic_mask = False
        self.loss_weight_photo = 1.0
        self.loss_weight_geometry = 0.5
        self.total_loss_weight = loss_weight
        self.data_type = data_type


    def photo_and_geometry_loss(self, tgt_img, ref_imgs, tgt_depth, ref_depths, intrinsics, poses, poses_inv):

        diff_img_list = []
        diff_color_list = []
        diff_depth_list = []
        valid_mask_list = []
        auto_mask_list = []

        for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
            diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1, auto_mask_tmp1 = self.compute_pairwise_loss(
                tgt_img, ref_img, tgt_depth,
                ref_depth, pose, intrinsics,
            )
            diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2, auto_mask_tmp2 = self.compute_pairwise_loss(
                ref_img, tgt_img, ref_depth,
                tgt_depth, pose_inv, intrinsics,
            )
            diff_img_list += [diff_img_tmp1, diff_img_tmp2]
            diff_color_list += [diff_color_tmp1, diff_color_tmp2]
            diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
            valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]
            auto_mask_list += [auto_mask_tmp1, auto_mask_tmp2]

        diff_img = torch.cat(diff_img_list, dim=1)
        diff_color = torch.cat(diff_color_list, dim=1)
        diff_depth = torch.cat(diff_depth_list, dim=1)
        valid_mask = torch.cat(valid_mask_list, dim=1)
        auto_mask = torch.cat(auto_mask_list, dim=1)

        # using photo loss to select best match in multiple views
        if not self.no_min_optimize:
            indices = torch.argmin(diff_color, dim=1, keepdim=True)

            diff_img = torch.gather(diff_img, 1, indices)
            diff_depth = torch.gather(diff_depth, 1, indices)
            valid_mask = torch.gather(valid_mask, 1, indices)
            auto_mask = torch.gather(auto_mask, 1, indices)

        if not self.no_auto_mask:
            photo_loss = self.mean_on_mask(diff_img, valid_mask * auto_mask)
            geometry_loss = self.mean_on_mask(diff_depth, valid_mask * auto_mask)
        else:
            photo_loss = self.mean_on_mask(diff_img, valid_mask)
            geometry_loss = self.mean_on_mask(diff_depth, valid_mask)

        dynamic_mask = None
        if self.return_dynamic_mask:
            # get dynamic mask for tgt image
            dynamic_mask_list = []
            for i in range(0, len(diff_depth_list), 2):
                tmp = diff_depth_list[i]
                tmp[valid_mask_list[i] < 1] = 0
                dynamic_mask_list += [1-tmp]

            dynamic_mask = torch.cat(dynamic_mask_list, dim=1).mean(dim=1, keepdim=True)

        return photo_loss, geometry_loss, dynamic_mask


    def compute_pairwise_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic):

        ref_img_warped, projected_depth, computed_depth = inverse_warp2(
            ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode='zeros')

        diff_depth = (computed_depth-projected_depth).abs() / \
            (computed_depth+projected_depth)

        # masking zero values
        valid_mask_ref = (ref_img_warped.abs().mean(
            dim=1, keepdim=True) > 1e-3).float()
        valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask = valid_mask_tgt * valid_mask_ref

        diff_color = (tgt_img-ref_img_warped).abs().mean(dim=1, keepdim=True)
        identity_warp_err = (tgt_img-ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()

        diff_img = (tgt_img-ref_img_warped).abs().clamp(0, 1)
        if not self.no_ssim:
            ssim_map = self.ssim_loss(tgt_img, ref_img_warped)
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        diff_img = torch.mean(diff_img, dim=1, keepdim=True)

        # reduce photometric loss weight for dynamic regions
        if not self.no_dynamic_mask:
            weight_mask = (1-diff_depth)
            diff_img = diff_img * weight_mask

        return diff_img, diff_color, diff_depth, valid_mask, auto_mask


    # compute mean value on a binary mask
    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        # if mask.sum() > 100:
        #     mean_value = (diff * mask).sum() / mask.sum()
        # else:
        #     mean_value = torch.tensor(0).float().to(device)
        mean_value = (diff * mask).sum() / (mask.sum() + 1e-6)
        return mean_value

    def forward(self, input, ref_input, prediction, ref_prediction, intrinsic, **kwargs):
        photo_loss, geometric_loss, dynamic_mask = self.photo_and_geometry_loss(
            tgt_img=input,
            ref_imgs=ref_input,
            tgt_depth=prediction,
            ref_depths=ref_prediction,
            intrinsics=intrinsic,
            poses=kwargs['pose'],
            poses_inv=kwargs['inv_pose']
        )
        loss = self.loss_weight_geometry * geometric_loss + self.loss_weight_photo * photo_loss
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'VNL error, {loss}')
        return loss * self.total_loss_weight
    