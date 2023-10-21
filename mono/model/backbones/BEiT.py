import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import timm
import numpy as np

from typing import Optional
from timm.models.beit import gen_relative_position_index

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

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)

        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

def make_backbone_default(
        model,
        features=[96, 192, 384, 768],
        size=[384, 384],
        hooks=[2, 5, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
        start_index_readout=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index_readout)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    return pretrained

def patch_embed_forward(self, x):
    """
    Modification of timm.models.layers.patch_embed.py: PatchEmbed.forward to support arbitrary window sizes.
    """
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)
    return x

def beit_forward_features(self, x):
    """
    Modification of timm.models.beit.py: Beit.forward_features to support arbitrary window sizes.
    """
    resolution = x.shape[2:]

    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    for blk in self.blocks:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
        else:
            x = blk(x, resolution, shared_rel_pos_bias=rel_pos_bias)
    x = self.norm(x)
    return x

def _get_rel_pos_bias(self, window_size):
    """
    Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
    """
    old_height = 2 * self.window_size[0] - 1
    old_width = 2 * self.window_size[1] - 1

    new_height = 2 * window_size[0] - 1
    new_width = 2 * window_size[1] - 1

    old_relative_position_bias_table = self.relative_position_bias_table

    old_num_relative_distance = self.num_relative_distance
    new_num_relative_distance = new_height * new_width + 3

    old_sub_table = old_relative_position_bias_table[:old_num_relative_distance - 3]

    old_sub_table = old_sub_table.reshape(1, old_width, old_height, -1).permute(0, 3, 1, 2)
    new_sub_table = F.interpolate(old_sub_table, size=(new_height, new_width), mode="bilinear")
    new_sub_table = new_sub_table.permute(0, 2, 3, 1).reshape(new_num_relative_distance - 3, -1)

    new_relative_position_bias_table = torch.cat(
        [new_sub_table, old_relative_position_bias_table[old_num_relative_distance - 3:]])

    key = str(window_size[1]) + "," + str(window_size[0])
    if key not in self.relative_position_indices.keys():
        self.relative_position_indices[key] = gen_relative_position_index(window_size)

    relative_position_bias = new_relative_position_bias_table[
        self.relative_position_indices[key].view(-1)].view(
        window_size[0] * window_size[1] + 1,
        window_size[0] * window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    return relative_position_bias.unsqueeze(0)

def attention_forward(self, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
    """
    Modification of timm.models.beit.py: Attention.forward to support arbitrary window sizes.
    """
    B, N, C = x.shape

    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    if self.relative_position_bias_table is not None:
        window_size = tuple(np.array(resolution) // 16)
        attn = attn + self._get_rel_pos_bias(window_size)
    if shared_rel_pos_bias is not None:
        attn = attn + shared_rel_pos_bias

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def block_forward(self, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
    """
    Modification of timm.models.beit.py: Block.forward to support arbitrary window sizes.
    """
    if self.gamma_1 is None:
        x = x + self.drop_path(self.attn(self.norm1(x), resolution, shared_rel_pos_bias=shared_rel_pos_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    else:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), resolution,
                                                        shared_rel_pos_bias=shared_rel_pos_bias))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    return x

def _make_beit_backbone(
        model,
        features=[96, 192, 384, 768],
        size=[384, 384],
        hooks=[0, 4, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
        start_index_readout=1,
):
    backbone = make_backbone_default(model, features, size, hooks, vit_features, use_readout, start_index,
                                     start_index_readout)

    backbone.model.patch_embed.forward = types.MethodType(patch_embed_forward, backbone.model.patch_embed)
    backbone.model.forward_features = types.MethodType(beit_forward_features, backbone.model)

    for block in backbone.model.blocks:
        attn = block.attn
        attn._get_rel_pos_bias = types.MethodType(_get_rel_pos_bias, attn)
        attn.forward = types.MethodType(attention_forward, attn)
        attn.relative_position_indices = {}

        block.forward = types.MethodType(block_forward, block)

    return backbone

def forward_adapted_unflatten(pretrained, x, function_name="forward_features"):
    b, c, h, w = x.shape

    exec(f"glob = pretrained.model.{function_name}(x)")

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3: len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3: len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3: len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3: len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4

class BEiT(nn.Module):
    def __init__(
        self,
        features=256,
        input_size=[512, 512],
        backbone="beit_large_patch16_512",
        readout="project",
        channels_last=False,
        use_bn=False,
        hooks=[5, 11, 17, 23],
        **kwargs
    ):

        super(BEiT, self).__init__()

        self.channels_last = channels_last

        # Instantiate backbone and reassemble blocks
        model = timm.create_model(backbone, pretrained=False)
        self.pretrained = _make_beit_backbone(
            model,
            features=[256, 512, 1024, 1024],
            size=input_size,
            vit_features=1024,
            hooks=hooks,
            use_readout=readout,
        )
        self.scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=1, expand=False
        )  # BEiT_512-L (backbone)

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

        self.forward_transformer = forward_adapted_unflatten


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

def beit_large_patch16_512(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BEiT(backbone="beit_large_patch16_512", input_size=[512, 512], **kwargs)
    if pretrained:
        ckpt_path = kwargs['checkpoint']
        model = load_pretrained_model(model, ckpt_path)

    return model

# def beit_large_patch16_384(pretrained=True, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = BEiT(backbone="beit_large_patch16_384", input_size=[384, 384], **kwargs)
#     if pretrained:
#         ckpt_path = kwargs['checkpoint']
#         model = load_pretrained_model(model, ckpt_path)

#     return model
    