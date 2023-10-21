
data_info={}

data_info['checkpoint']={
    'database_root': '/mnt/nas/share/home/xugk/ckpt_files',

    # pretrained weight for convnext
    'convnext_large': 'convnext/convnext_large_22k_1k_384.pth',

    # NOTE: disabled here for beit and swin2. Please use load-from to load pretrained weight during training.
    'beit_large_patch16_512': '',
    'swinv2_large_window12to24_192to384_22kft1k': '',
}