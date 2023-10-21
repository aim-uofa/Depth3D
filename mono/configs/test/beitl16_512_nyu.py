_base_=[
        '../_base_/models/encoder_decoder/beitl16_512.dpthead.py',
        '../_base_/datasets/_data_base_.py',

        '../_base_/datasets/nyu.py',
        
        # debug
        '../_base_/losses/all_losses.py',
       ]

# loss method
losses=dict(
    decoder_losses=[
        dict(type='VNLoss', sample_ratio=0.2, loss_weight=1.0),
        dict(type='L1Loss', loss_weight=1.0),
        dict(type='SkyRegularizationLoss', loss_weight=0.001, sample_ratio=0.2, regress_value=250),
        dict(type='HDNRandomLoss', loss_weight=2, random_num=10),
        dict(type='HDSNRandomLoss', loss_weight=2, random_num=20, batch_limit=4),
        dict(type='EdgeguidedNormalLoss', loss_weight=1),
        dict(type='PWNPlanesLoss', loss_weight=1),
        dict(type='ConfidenceLoss', loss_weight=1),
],
)

# data configs, some similar data are merged together
data_array = [
    [
        dict(NYU='NYU_dataset'),
    ],
]

model = dict(
    backbone=dict(
        pretrained=False,
    )
)

# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        img_size=(512, 512),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 150),
    crop_size = (512, 512),
    disp_pred = True,
    disp_scale = 1000,
    disp_min = 1 / 300,
) 

dist_params = dict(
    port=None,
    backend='nccl',
    dist_url='env://',
    nnodes=1,
    node_rank=0,
    num_gpus_per_node=1,
    world_size=1,
    global_rank=0,
)

# online evaluation
evaluation = dict(online_eval=True, interval=2500, metrics=['abs_rel', 'delta1'])

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
checkpoint_config = dict(by_epoch=False, interval=2500)
runner = dict(type='IterBasedRunner', max_iters=400000)

# optimizer
optimizer = dict(
    type='AdamW', 
    encoder=dict(lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    decoder=dict(lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
)
# schedule
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-6, by_epoch=False)

batchsize_per_gpu = 2
thread_per_gpu = 4

NYU_dataset=dict(
    data=dict(
        # configs for the training pipeline
        test=dict(
            pipeline=[dict(type='BGR2RGB'),
                        dict(type='LableScaleCanonical'),
                        dict(type='ResizeKeepRatio', 
                        resize_size=(512, 512),
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                        dict(type='ToTensor'),
                        dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                        ],),),
)
