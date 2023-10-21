# dataset settings

GraspNet_dataset=dict(
    lib = 'GraspNetDataset',
#     data_root = 'data/public_datasets',
    data_name = 'GraspNet',
    transfer_to_canonical = True,    
    metric_scale = 256.0, # NOTE: unknown now, regard it as scale-invariant depth
#     original_focal_length = 518.857,
#     original_size = (480, 640),
    data_type='sfm',
    data = dict(
    # configs for the training pipeline
    train=dict(
        sample_ratio = 1.0,
        sample_size = -1,
        pipeline=[dict(type='BGR2RGB'),
                dict(type='LableScaleCanonical'),
                dict(type='RandomResize',
                        prob=0.5,
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
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                ],
    ),

    # configs for the val pipeline
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                dict(type='LableScaleCanonical'),
                dict(type='ResizeKeepRatio', 
                    resize_size=(544, 1216),
                    ignore_label=-1, 
                    padding=[0, 0, 0]),
                dict(type='ToTensor'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                ],
        sample_ratio = 1.0,
        sample_size = 100,
    ),
    # configs for the training pipeline
    test=dict(
          anno_path='',
          pipeline=[dict(type='BGR2RGB'),
                    dict(type='LableScaleCanonical'),
                    dict(type='ResizeKeepRatio', 
                       resize_size=(544, 1216),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),

                    # dict(type='Resize', 
                    #    resize_size=(544, 1216),
                    #    ignore_label=-1, 
                    #    padding=[0, 0, 0]),

                    # dict(type='RandomCrop', 
                    #     crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                    #     crop_type='rand', 
                    #     ignore_label=-1, 
                    #     padding=[0, 0, 0]),
                    dict(type='ToTensor'),
                    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                    ],
        sample_ratio = 1.0,
        sample_size = -1,),
     ),
)