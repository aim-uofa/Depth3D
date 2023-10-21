# model settings
_base_ = ['../backbones/swin2l24_384.py',]
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='DPTHead',
        features=256,
        non_negative=True,
        hooks=[1, 1, 17, 1],
        use_bn=False,
        prefix='decode_heads.'),
)
