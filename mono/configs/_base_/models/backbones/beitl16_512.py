#_base_ = ['./_model_base_.py',]

#'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'

model = dict(
    #type='EncoderDecoderAuxi',
    backbone=dict(
        type='beit_large_patch16_512',
        pretrained=False,
        features=256,
        readout="project",
        channels_last=False,
        use_bn=False,
        hooks=[5, 11, 17, 23],
        checkpoint='...',
        prefix='backbones.',
    ),
)
