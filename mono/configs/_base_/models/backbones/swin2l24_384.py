#_base_ = ['./_model_base_.py',]

#'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'

model = dict(
    #type='EncoderDecoderAuxi',
    backbone=dict(
        type='swinv2_large_window12to24_192to384_22kft1k',
        pretrained=False,
        features=256,
        readout="project",
        channels_last=False,
        use_bn=False,
        hooks=[1, 1, 17, 1],
        checkpoint='...',
        prefix='backbones.',
    ),
)
