if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/TUM'
    split_root = '/mnt/nas/share/home/xugk/data/TUM/'


    files = []

    for scene_name in os.listdir(data_root):
        print(scene_name)
        scene_root = osp.join(data_root, scene_name)
        if not (osp.isdir(scene_root) and osp.exists(scene_root)):
            continue
        
        info_path = osp.join(scene_root, 'rgb_depth_pose.txt')
        with open(info_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            rgb_name = line.split(' ')[1]
            depth_name = line.split(' ')[3]

            img_path = osp.join(scene_root, rgb_name)
            depth_path = osp.join(scene_root, depth_name)
            cam_in = [525.0, 525.0, 319.5, 239.5]

            meta_data = {}
            meta_data['cam_in'] = cam_in
            meta_data['rgb'] = img_path.split(split_root)[-1]
            meta_data['depth'] = depth_path.split(split_root)[-1]
            files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/TUM/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)