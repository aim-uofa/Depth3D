import torch
import torch.nn as nn
from mono.utils.comm import get_func
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mono.utils.transform import gray_to_colormap


class BaseDepthModel(nn.Module):
    def __init__(self, cfg, criterions, **kwargs) -> None:
        super(BaseDepthModel, self).__init__()
        model_type = cfg.model.type
        self.depth_model = get_func('mono.model.model_pipelines.' + model_type)(cfg)

        self.criterions_main = criterions['decoder_losses'] if criterions and 'decoder_losses' in criterions else None

        self.training = True

    def forward(self, data):
        if self.training:
            data['training'] = True
        else:
            data['training'] = False
        output = self.depth_model(**data)

        losses_dict = {}
        if self.training:
            output.update(data)
            losses_dict = self.get_loss(output)
        return output['prediction'], losses_dict, output['confidence'], output
    
    def get_loss(self, paras):
        losses_dict = {}
        # Losses for training
        if self.training:
            # decode branch
            losses_dict.update(self.compute_decoder_loss(paras))

            total_loss = sum(losses_dict.values())
            losses_dict['total_loss'] = total_loss
        return losses_dict
    
    def compute_decoder_loss(self, paras):
        losses_dict = {}
        decode_loss_dict = self.branch_loss(
            criterions=self.criterions_main,
            branch='decode',
            **paras
        )
        return decode_loss_dict
    
    def branch_loss(self, prediction, pred_logit, criterions, branch='decode', **kwargs):
        B, _, _, _ = prediction.shape
        losses_dict = {}
        args = dict(pred_logit=pred_logit)

        target = kwargs.pop('target')
        args.update(kwargs)

        # data type for each batch
        batches_data_type = np.array(kwargs['data_type'])
        batches_data_names = np.array(kwargs['dataset'])

        # resize the target
        if target.shape[2] != prediction.shape[2] and target.shape[3] != prediction.shape[3]:
            _, _, H, W = prediction.shape
            target = nn.functional.interpolate(target, (H,W), mode='nearest')

        mask = target > 1e-8
        for loss_method in criterions:
            # sample batches, which satisfy the loss requirement for data types
            data_type_req = np.array(loss_method.data_type)[:, None]
            batch_mask = torch.from_numpy(np.any(data_type_req == batches_data_type, axis=0)).cuda()
            new_mask = mask * batch_mask[:, None, None, None]

            loss_tmp = loss_method(
                prediction=prediction,
                target=target,
                mask=new_mask,
                **args,
            )
            losses_dict[branch + '_' + loss_method._get_name()] = loss_tmp
        return losses_dict
    
    def mask_batches(self, prediction, target, mask, batches_data_names, data_type_req):
        batch_mask = np.any(data_type_req == batches_data_names, axis=0)
        prediction = prediction[batch_mask]
        target = target[batch_mask]
        mask = mask[batch_mask]
        return prediction, target, mask, batch_mask

    def inference(self, data):
        with torch.no_grad():
            pred_depth, _, confidence = self.forward(data)
        return pred_depth, confidence
    
def min_pool2d(tensor, kernel, stride=1):
    tensor = tensor * -1.0
    tensor = F.max_pool2d(tensor, kernel, padding=kernel // 2, stride=stride)
    tensor = -1.0 * tensor
    return tensor
