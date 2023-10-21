# canonical camera setting and basic data setting
data_basic=dict(
    canonical_space = dict(
        img_size=(544, 1216),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.01, 150),
    crop_size = (544, 1216),
    clip_depth_range=(0.9, 150),
) 
