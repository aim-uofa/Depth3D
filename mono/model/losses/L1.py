import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mono.utils.do_test import align_scale

class L1Loss(nn.Module):
    '''
    Compute L1 Loss.
    '''
    def __init__(self, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def forward(self, prediction, target, mask=None, **kwargs):

        if self.data_type == 'sfm': # gt depth is scale-invariant, we align target with prediction
            # target, _ = align_scale(target, prediction)
            return prediction.sum() * 0.
        
        # prediction_0 = prediction[0,0].detach().cpu().numpy()
        # target_0 = target[0,0].detach().cpu().numpy()
        # mask_0 = mask[0,0].detach().cpu().numpy()
        # # print('prediction_0 :', prediction_0.shape)
        # # print('target_0:' , target_0.shape)
        # # print('mask_0 :', mask_0.shape)
        # plt.imsave('temp_prediction_0.png', prediction_0, cmap='rainbow')
        # plt.imsave('temp_target_0.png', target_0, cmap='rainbow')
        # plt.imsave('temp_mask_0.png', mask_0, cmap='rainbow')

        diff = torch.abs(prediction[mask] - target[mask])
        loss = torch.sum(diff) / (diff.numel() + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
        
        return loss * self.loss_weight
    