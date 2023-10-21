import torch
import torch.nn as nn

from mono.utils.do_test import align_scale

class SSILoss(nn.Module):
    """
    Scale shift invariant MAE loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(SSILoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def ssi_mae(self, target, prediction, mask):
        valid_pixes = torch.sum(mask) + self.eps

        gt_median = torch.median(target) if target.numel() else 0
        gt_s = torch.abs(target - gt_median).sum() / valid_pixes
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = torch.median(prediction) if prediction.numel() else 0
        pred_s = torch.abs(prediction - pred_median).sum() / valid_pixes
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)
        
        ssi_mae_sum = torch.sum(torch.abs(gt_trans - pred_trans))
        return ssi_mae_sum, valid_pixes

    def forward(self, prediction, target, mask=None, **kwargs):
        """
        Calculate loss.
        """

        if self.data_type == 'sfm': # gt depth is scale-invariant, we align target with prediction
            target, _ = align_scale(target, prediction)

        B, C, H, W = prediction.shape
        loss = 0
        valid_pix = 0
        for i in range(B):
            mask_i = mask[i, ...]
            gt_depth_i = target[i, ...][mask_i]
            pred_depth_i = prediction[i, ...][mask_i]
            ssi_sum, valid_pix_i = self.ssi_mae(pred_depth_i, gt_depth_i, mask_i)
            loss += ssi_sum
            valid_pix += valid_pix_i
        loss /= (valid_pix + self.eps)
        return loss * self.loss_weight


if __name__ == '__main__':
    ssimae_loss = SSILoss()
    pred_depth = torch.rand([3, 1, 385, 513]).cuda()
    gt_depth = torch.rand([3, 1, 385, 513]).cuda()

    loss = ssimae_loss(pred_depth, gt_depth)
    print(loss)
