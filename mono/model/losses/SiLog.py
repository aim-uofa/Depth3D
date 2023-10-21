import torch
import torch.nn as nn

from mono.utils.do_test import align_scale

class SilogLoss(nn.Module):
    def __init__(self, variance_focus=0.5, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(SilogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask=None, **kwargs):

        if self.data_type == 'sfm': # gt depth is scale-invariant, we align target with prediction
            target, _ = align_scale(target, prediction)

        d = torch.log(prediction[mask]) - torch.log(target[mask])
        d_square_mean = torch.sum(d ** 2) / (d.numel() + self.eps)
        d_mean = torch.sum(d) / (d.numel() + self.eps)
        loss = d_square_mean - self.variance_focus * (d_mean ** 2)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'SilogLoss error, {loss}')
        return loss * self.loss_weight
      
if __name__ == '__main__':
    silog = SilogLoss()
    pred = torch.rand((2, 3, 256, 256)).cuda()
    gt = torch.rand((2, 3, 256, 256)).cuda()
    mask = gt > 0
    out = silog(pred, gt, mask)
    print(out)
