if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/ibims/ibims/ibims1_core_raw/'
    split_root = '/mnt/nas/share/home/xugk/data/ibims/'
    rgb_root = osp.join(data_root, 'rgb')

    files = []
    for file_name in os.listdir(rgb_root):
        if not file_name.endswith('.png'):
            continue
        print('file_name :', file_name)
        
        rgb_path = osp.join(rgb_root, file_name)
        depth_path = rgb_path.replace('/rgb/', '/depth/')
        depth_mask_path = rgb_path.replace('/rgb/', '/mask_invalid/')
        if (not osp.exists(depth_path)) or (not osp.exists(depth_mask_path)):
            assert False
        
        cam_in_path = rgb_path.replace('/rgb/', '/calib/').replace('.png', '.txt')
        with open(cam_in_path, 'r') as f:
            lines = f.readlines()
        
        fx, fy, cx, cy = lines[0].strip().split(',')
        cam_in = [float(fx), float(fy), float(cx), float(cy)]
        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path.split(split_root)[-1]
        meta_data['depth'] = depth_path.split(split_root)[-1]
        meta_data['depth_mask'] = depth_mask_path.split(split_root)[-1]

        files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/ibims/test_annotations.json', 'w') as f:
        json.dump(files_dict, f)