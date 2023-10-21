import torch
import torch.nn as nn

from mono.utils.do_test import align_scale

class ConfidenceLoss(nn.Module):
    '''
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.
    '''
    def __init__(self, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(ConfidenceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def forward(self, prediction, target, confidence, mask=None, **kwargs):

        if self.data_type == 'sfm': # gt depth is scale-invariant, we align target with prediction
            target, _ = align_scale(target, prediction)

        conf_mask = torch.abs(target - prediction) < target
        conf_mask = conf_mask & mask
        gt_confidence = 1 - torch.abs((prediction[conf_mask] - target[conf_mask]) / target[conf_mask])
        loss = torch.sum(torch.abs(confidence[conf_mask] - gt_confidence)) / (confidence[conf_mask].numel() + self.eps)

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
        return loss * self.loss_weight
    