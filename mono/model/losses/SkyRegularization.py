import numpy as np
import torch
import torch.nn as nn

from mono.utils.do_test import align_scale

class SkyRegularizationLoss(nn.Module):
    """
    Enforce losses on pixels without any gts.
    """
    def __init__(self, loss_weight=0.1, data_type=['sfm', 'stereo', 'lidar'], sky_id=142, sample_ratio=0.4, regress_value=1.8, **kwargs):
        super(SkyRegularizationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.sky_id = sky_id
        self.sample_ratio = sample_ratio
        self.eps = 1e-6
        self.regress_value = regress_value

        if 'disp_pred' in kwargs:
            self.disp_pred = kwargs['disp_pred']
        else:
            self.disp_pred = False
    
    def loss1(self, pred_sky):
        loss = 1/ torch.exp((torch.sum(pred_sky) / (pred_sky.numel() + self.eps)))
        return loss

    def loss2(self, pred_sky):
        loss = torch.sum(torch.abs(pred_sky - self.regress_value)) / (pred_sky.numel() + self.eps)
        return loss
    
    def loss_disp(self, pred_disp_sky):
        loss = torch.sum(torch.abs(pred_disp_sky - 0)) / (pred_disp_sky.numel() + self.eps)
        return loss

    def forward(self, prediction, target, mask=None, sem_mask=None, **kwargs):
        
        if not self.disp_pred:
            if self.data_type == 'sfm': # gt depth is scale-invariant, we align target with prediction
                target, _ = align_scale(target, prediction)

            sky_mask = sem_mask == self.sky_id
            pred_sky = prediction[sky_mask]
            pred_sky_numel = pred_sky.numel()

            
            if pred_sky.numel() > 50:
                samples = np.random.choice(pred_sky_numel, int(pred_sky_numel*self.sample_ratio), replace=False)
                pred_sky = pred_sky[samples]
            #loss = - torch.sum(pred_wo_gt) / (pred_wo_gt.numel() + 1e-8)
            loss = self.loss2(pred_sky)
            # if torch.isnan(loss).item() | torch.isinf(loss).item():
            #     # raise RuntimeError(f'Sky Loss error, {loss}')
            #     loss = torch.sum(prediction) * 0.0
            return loss * self.loss_weight
        else:
            # predict disparity
            sky_mask = sem_mask == self.sky_id
            pred_disp = kwargs['pred_disp']
            pred_sky = pred_disp[sky_mask]
            pred_sky_numel = pred_sky.numel()
            
            if pred_sky.numel() > 50:
                samples = np.random.choice(pred_sky_numel, int(pred_sky_numel*self.sample_ratio), replace=False)
                pred_sky = pred_sky[samples]
            loss = self.loss_disp(pred_sky)
            return loss * self.loss_weight

if __name__ == '__main__':
    import cv2
    sky = SkyRegularizationLoss()
    pred_depth = np.random.random([2, 1, 480, 640])
    gt_depth = np.zeros_like(pred_depth) #np.random.random([2, 1, 480, 640])
    intrinsic = [[[100, 0, 200], [0, 100, 200], [0, 0, 1]], [[100, 0, 200], [0, 100, 200], [0, 0, 1]],]
    gt_depth = torch.tensor(np.array(gt_depth, np.float32)).cuda()