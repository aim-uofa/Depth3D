_base_=['../_base_/losses/all_losses.py',
       '../_base_/models/encoder_decoder/beitl16_512.dpthead.py',

       '../_base_/datasets/ddad.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/datasets/argovers2.py',
       '../_base_/datasets/cityscapes.py',
       '../_base_/datasets/dsec.py',
       '../_base_/datasets/lyft.py',
       '../_base_/datasets/mapillary_psd.py',
       '../_base_/datasets/diml.py',
       '../_base_/datasets/taskonomy.py',
       '../_base_/datasets/uasol.py',
       '../_base_/datasets/pandaset.py',
       '../_base_/datasets/waymo.py',

       '../_base_/datasets/avd.py',
       '../_base_/datasets/blendedmvs.py',
       '../_base_/datasets/diode.py',
       '../_base_/datasets/graspnet.py',
       '../_base_/datasets/hypersim.py',
       '../_base_/datasets/kitti.py',
       '../_base_/datasets/nyu.py',
       '../_base_/datasets/tartanair.py',
       '../_base_/datasets/tum.py',
       '../_base_/datasets/scannet.py',

       '../_base_/default_runtime.py',
       '../_base_/schedules/schedule_1m.py',
       
       ]


# loss method
losses=dict(
    decoder_losses=[
        dict(type='VNLoss', loss_weight=1.0, sample_ratio=0.2),
        dict(type='L1Loss', loss_weight=1.0),
        dict(type='SkyRegularizationLoss', loss_weight=0.01, sample_ratio=0.2, regress_value=250, disp_pred=True),
        dict(type='HDNRandomLoss', loss_weight=1, random_num=10),
        dict(type='HDSNRandomLoss', loss_weight=1, random_num=20, batch_limit=4),
        dict(type='EdgeguidedNormalLoss', loss_weight=1),
        dict(type='PWNPlanesLoss', loss_weight=2),
        # dict(type='ConfidenceLoss', loss_weight=1),
],
)

# data configs, some similar data are merged together
data_array = [
    # group 4 pseudo 
    [
        dict(UASOL='UASOL_dataset'),
    ],

    [
        dict(Cityscapes='Cityscapes_dataset'),
    ],
    
    [
        dict(DIML='DIML_dataset'),
    ],
    
    [
        dict(KITTI='KITTI_dataset'),
    ],
    
    [
        dict(Lyft='Lyft_dataset'),
    ],
    
    [
        dict(DDAD='DDAD_dataset'),
    ],
    
    [
        dict(Pandaset='Pandaset_dataset'),
    ],
    
    [
        dict(Waymo='Waymo_dataset'),
    ],

    [
        dict(Argovers2='Argovers2_dataset'),
    ],

    [
        dict(Mapillary_PSD='MapillaryPSD_dataset'),
    ],

    # group 7
    [
        dict(Taskonomy='Taskonomy_dataset')  
    ],

    # group 1
    [
        dict(DSEC='DSEC_dataset'),
    ],

    [
        dict(DIODE='DIODE_dataset'),
    ],
    
    [
        dict(Hypersim='Hypersim_dataset'),
    ],

    [
        dict(Tartanair='Tartanair_dataset'),
    ],
    
    [
        dict(GraspNet='GraspNet_dataset'),
    ],

    [
        dict(BlendedMVS='BlendedMVS_dataset'),
    ],

    [
        dict(AVD='AVD_dataset'),
    ],

    [
        dict(NYU='NYU_dataset'),
    ],

    [
        dict(TUM='TUM_dataset'),
    ],

    [
        dict(Scannet='Scannet_dataset'),
    ],
]

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

# online evaluation
evaluation = dict(online_eval=True, interval=10000, metrics=['abs_rel', 'delta1'])

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
checkpoint_config = dict(by_epoch=False, interval=10000)
runner = dict(type='IterBasedRunner', max_iters=400000)

# optimizer
optimizer = dict(
    type='AdamW', 
    encoder=dict(key_rank=0, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    decoder=dict(key_rank=1, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
)
# schedule
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-8, by_epoch=False)

batchsize_per_gpu = 16
thread_per_gpu = 12
Argovers2_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the training pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)


Cityscapes_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the training pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

DIML_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the training pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

Lyft_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the training pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

DDAD_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[# dict(type='BGR2RGB'), # NOTE: check RGB img
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[# dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),

        # configs for the test pipeline
        test=dict(
            anno_path='DDAD/annotations/test_annotations.json',
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_ratio = 1.0,
            sample_size = 100,
        ),
    )
)

DSEC_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

MapillaryPSD_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 1,
        ),
    )
)

Pandaset_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

Taskonomy_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            ratio_range=(0.85, 1.15),
                            is_lidar=False),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

UASOL_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

Waymo_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

AVD_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

BlendedMVS_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

DIODE_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

GraspNet_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

Hypersim_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

KITTI_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

NYU_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

Tartanair_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

TUM_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)

Scannet_dataset=dict(
    data = dict(
        # configs for the training pipeline
        train=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='RandomResize',
                            prob=0.5,
                            ratio_range=(0.85, 1.15),
                            is_lidar=True),
                    dict(type='RandomCrop', 
                        crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                        crop_type='rand', 
                        ignore_label=-1, 
                        padding=[0, 0, 0]),
                    dict(type='RandomEdgeMask',
                            mask_maxsize=50,
                            prob=0.2,
                            rgb_invalid=[0,0,0],
                            label_invalid=-1,), 
                    dict(type='RandomHorizontalFlip', 
                        prob=0.4),
                    dict(type='PhotoMetricDistortion', 
                        to_gray_prob=0.1,
                        distortion_prob=0.1,),
                    dict(type='Weather',
                        prob=0.05),
                    dict(type='RandomBlur', 
                        prob=0.05),
                    dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],),

        # configs for the val pipeline
        val=dict(
            pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(512, 512),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
                    ],
            sample_size = 100,
        ),
    )
)