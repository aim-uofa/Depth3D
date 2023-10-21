# model settings
_base_ = ['../backbones/beitl16_512.py',]
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='DPTHead',
        features=256,
        non_negative=True,
        hooks=[5, 11, 17, 23],
        use_bn=False,
        prefix='decode_heads.'),
)
