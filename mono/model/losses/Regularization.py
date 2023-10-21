import torch
import torch.nn as nn

from mono.utils.do_test import align_scale

class RegularizationLoss(nn.Module):
    '''
    Enforce losses on pixels without any gts.
    '''
    def __init__(self, loss_weight=0.1, data_type=['sfm', 'stereo', 'lidar'], **kwargs) -> None:
        super(RegularizationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask=None, **kwargs):

        if self.data_type == 'sfm': # gt depth is scale-invariant, we align target with prediction
            target, _ = align_scale(target, prediction)

        pred_wo_gt = prediction[~mask]
        loss = 1 / (torch.sum(pred_wo_gt) / (pred_wo_gt.numel() + self.eps))
        return loss * self.loss_weight
    