if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/ETH3D/ETH3D/'
    split_root = '/mnt/nas/share/home/xugk/data/ETH3D/'

    files = []
    for folder_name in os.listdir(data_root):
        if '_depth' in folder_name:
            continue
        if '_dslr_jpg' not in folder_name:
            continue
        
        scene_name = folder_name.split('_dslr_jpg')[0]
        print(scene_name)

        rgb_root = osp.join(data_root, folder_name, scene_name, 'images', 'dslr_images')
        if not (osp.isdir(rgb_root) and osp.exists(rgb_root)):
            continue
        
        for file_name in os.listdir(rgb_root):
            if not file_name.endswith('.JPG'):
                continue

            rgb_path = osp.join(rgb_root, file_name)
            depth_path = osp.join(data_root, folder_name.replace('_dslr_jpg', '_dslr_depth'), scene_name, 'ground_truth_depth', 'dslr_images', file_name)
            if not osp.exists(depth_path):
                assert False
            
            cam_in_path = osp.join(data_root, scene_name, 'dslr_calibration_undistorted', 'cameras.txt')
            with open(cam_in_path, 'r') as f:
                lines = f.readlines()
            
            fx, fy, cx, cy = lines[-1].strip().split(' ')[-4:]
            cam_in = [float(fx), float(fy), float(cx), float(cy)]
            meta_data = {}
            meta_data['cam_in'] = cam_in
            meta_data['rgb'] = rgb_path.split(split_root)[-1]
            meta_data['depth'] = depth_path.split(split_root)[-1]
            files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/ETH3D/test_annotations.json', 'w') as f:
        json.dump(files_dict, f)