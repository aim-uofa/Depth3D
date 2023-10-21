# dataset settings
# data will resized/cropped to the canonical size, refer to ._data_base_.py

Taskonomy_dataset=dict(
    lib = 'TaskonomyDataset',
    data_root = 'data/public_datasets',
    data_name = 'Taskonomy',
    transfer_to_canonical = True,
    metric_scale = 512.0,
    original_focal_length = 575.6656,
    original_size = (512, 512),
    data_type='stereo',
    data = dict(
    # configs for the training pipeline
    train=dict(
        anno_path='Taskonomy/annotations/test.json',
        sample_ratio = 1.0,
        sample_size = -1,
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='ResizeCanonical', ratio_range=(0.9, 1.4)),
                  dict(type='RandomCrop', 
                       crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                       crop_type='rand_in_field', 
                       ignore_label=-1, 
                       padding=[123.675, 116.28, 103.53]),
                  dict(type='RandomHorizontalFlip', 
                       prob=0.4),
                  dict(type='PhotoMetricDistortion', 
                       to_gray_prob=0.1,
                       distortion_prob=0.05,),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 50)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],),

    # configs for the training pipeline
    val=dict(
        anno_path='Taskonomy/annotations/test.json',
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='ResizeCanonical', ratio_range=(1.0, 1.0)),
                  dict(type='RandomCrop', 
                       crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                       crop_type='center', 
                       ignore_label=-1, 
                       padding=[123.675, 116.28, 103.53]),
               #    dict(type='AdjustSize', 
               #         ignore_label=-1, 
               #         padding=[123.675, 116.28, 103.53]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_ratio = 1.0,
        sample_size = 200,),
    # configs for the training pipeline
    test=dict(
        anno_path='Taskonomy/annotations/test.json',
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
        sample_size = 2000,),
     ),
)