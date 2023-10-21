import torch
import torch.nn as nn
from mono.utils.comm import get_func
from .model_pipelines.__base_model__ import BaseDepthModel

class DepthModel(BaseDepthModel):
    def __init__(self, cfg, criterions, **kwards):
        super(DepthModel, self).__init__(cfg, criterions)   
        model_type = cfg.model.type
        self.training = True
        
    def inference(self, data):
        with torch.no_grad():
            pred_depth, _, confidence, output_dict = self.forward(data)       
        return pred_depth, confidence, output_dict

def get_monodepth_model(
    cfg : dict,
    criterions: dict,
    **kwargs
    ) -> nn.Module:
    # config depth  model
    model = DepthModel(cfg, criterions, **kwargs)
    #model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model

def get_configured_monodepth_model(
    cfg: dict,
    criterions: dict,
    ) -> nn.Module:
    """
        Args:
        @ configs: configures for the network.
        @ load_imagenet_model: whether to initialize from ImageNet-pretrained model.
        @ imagenet_ckpt_fpath: string representing path to file with weights to initialize model with.
        Returns:
        # model: depth model.
    """
    model = get_monodepth_model(cfg, criterions)
    return model
