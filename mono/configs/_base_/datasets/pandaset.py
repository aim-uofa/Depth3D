# dataset settings

Pandaset_dataset=dict(
    lib = 'PandasetDataset',
    data_root = 'data/public_datasets',
    data_name = 'Pandaset',
    transfer_to_canonical = True,    
    metric_scale = 200.0,
    original_focal_length = (1970.01, 930.45, 929.84),
    original_size = (1080, 1920),
    data_type='lidar',
    data = dict(
    # configs for the training pipeline
    train=dict(
        anno_path='Pandaset/annotations/train_annotations.json',
        sample_ratio = 1.0,
        sample_size = -1,
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LiDarResizeCanonical', ratio_range=(0.9, 1.4)),
                  dict(type='RandomCrop', 
                       crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                       crop_type='rand_in_field', 
                       ignore_label=-1, 
                       padding=[123.675, 116.28, 103.53]),
                  dict(type='RandomHorizontalFlip', 
                       prob=0.4),
                  dict(type='PhotoMetricDistortion', 
                       to_gray_prob=0.2,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.1),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],),

    # configs for the training pipeline
    val=dict(
        anno_path='Pandaset/annotations/val_annotations.json',
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LiDarResizeCanonical', ratio_range=(1.0, 1.0)),
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
        sample_size = 100,),
    # configs for the training pipeline
    test=dict(
        anno_path='Pandaset/annotations/test_annotations.json',
        pipeline=[
               dict(type='BGR2RGB'),
               dict(type='LableScaleCanonical'),
               dict(type='ResizeKeepRatio', 
                    resize_size=(544, 1216),
                    ignore_label=-1, 
                    padding=[0, 0, 0]),
               dict(type='ToTensor'),
               dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
          ],
        sample_ratio = 1.0,
        sample_size = 800,),
    ),
)    
