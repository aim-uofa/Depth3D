import torch
import torch.nn as nn
from mono.utils.comm import get_func

class DensePredModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(DensePredModel, self).__init__()

        self.encoder = get_func('mono.model.' + cfg.model.backbone.prefix + cfg.model.backbone.type)(**cfg.model.backbone)
        self.decoder = get_func('mono.model.' + cfg.model.decode_head.prefix + cfg.model.decode_head.type)(cfg)

        self.training = True
        self.cfg = cfg

    def forward(self, input, **kwargs):
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        out = self.decoder(features, **kwargs)
        # print('out :', out['prediction'].shape)

        if 'disp_pred' in self.cfg.data_basic and self.cfg.data_basic.disp_pred == True:
            assert 'disp_scale' in self.cfg.data_basic
            disp_scale = self.cfg.data_basic.disp_scale
            disp_min = self.cfg.data_basic.disp_min
            pred_disp = out.pop('prediction')
            pred_depth = disp_scale / (pred_disp + disp_scale * disp_min)
            out['prediction'] = pred_depth
            out['pred_disp'] = pred_disp

            assert torch.all(pred_depth >= 0)
            assert torch.all(pred_disp >= 0)

            # print('pred_depth111 :', out['prediction'][kwargs['target'] != -1].view(-1)[1000])
            # print('pred_disp :', pred_disp.view(-1)[1000])
            # print('disp_scale :', disp_scale)
            # print('disp_min :', disp_min)

        return out
        