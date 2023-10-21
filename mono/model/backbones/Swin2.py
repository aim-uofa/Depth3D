import torch
import torch.nn as nn
import numpy as np

import timm

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def forward_default(pretrained, x, function_name="forward_features"):
    exec(f"pretrained.model.{function_name}(x)")

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    if hasattr(pretrained, "act_postprocess1"):
        layer_1 = pretrained.act_postprocess1(layer_1)
    if hasattr(pretrained, "act_postprocess2"):
        layer_2 = pretrained.act_postprocess2(layer_2)
    if hasattr(pretrained, "act_postprocess3"):
        layer_3 = pretrained.act_postprocess3(layer_3)
    if hasattr(pretrained, "act_postprocess4"):
        layer_4 = pretrained.act_postprocess4(layer_4)

    return layer_1, layer_2, layer_3, layer_4


def forward_swin(pretrained, x):
    return forward_default(pretrained, x)

def _make_swin_backbone(
        model,
        hooks=[1, 1, 17, 1],
        patch_grid=[96, 96]
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.layers[0].blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.layers[1].blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.layers[2].blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.layers[3].blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if hasattr(model, "patch_grid"):
        used_patch_grid = model.patch_grid
    else:
        used_patch_grid = patch_grid

    patch_grid_size = np.array(used_patch_grid, dtype=int)

    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))
    )
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 2).tolist()))
    )
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 4).tolist()))
    )
    pretrained.act_postprocess4 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 8).tolist()))
    )

    return pretrained


class Swin2(nn.Module):
    def __init__(
        self,
        features=256,
        input_size=[384, 384],
        backbone="swinv2_large_window12to24_192to384_22kft1k",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs
    ):

        super(Swin2, self).__init__()

        self.channels_last = channels_last

        # Instantiate backbone and reassemble blocks
        model = timm.create_model(backbone, pretrained=False)

        if backbone == 'swinv2_large_window12to24_192to384_22kft1k':
            hooks = [1, 1, 17, 1]
            scratch_channels = [192, 384, 768, 1536]
        else:
            raise ValueError


        self.pretrained = _make_swin_backbone(
            model,
            hooks=hooks,
        )
        self.scratch = _make_scratch(
            scratch_channels, features, groups=1, expand=False
        )

        # self.pretrained, self.scratch = _make_encoder(
        #     backbone,
        #     features,
        #     False, # Set to true of you want to train from scratch, uses ImageNet weights
        #     groups=1,
        #     expand=False,
        #     exportable=False,
        #     hooks=hooks,
        #     use_readout=readout,
        #     in_features=in_features,
        # )

        self.number_layers = len(hooks) if hooks is not None else 4

        self.forward_transformer = forward_swin


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers
        
        features = []

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        features.append(layer_1_rn)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        features.append(layer_2_rn)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        features.append(layer_3_rn)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            features.append(layer_4_rn)

        return features

def load_pretrained_model(model, ckpt_path):

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    #url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
    #checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {}
    unmatched_pretrained_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            unmatched_pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(
        'Successfully loaded pretrained %d params, and %d paras are unmatched.'
        %(len(pretrained_dict.keys()), len(unmatched_pretrained_dict.keys())))
    print('Unmatched pretrained paras are :', unmatched_pretrained_dict.keys())

    return model


def swinv2_large_window12to24_192to384_22kft1k(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Swin2(backbone="swinv2_large_window12to24_192to384_22kft1k", **kwargs)
    if pretrained:
        ckpt_path = kwargs['checkpoint']
        model = load_pretrained_model(model, ckpt_path)

    return model
