if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/7scenes/7scenes'
    split_root = '/mnt/nas/share/home/xugk/data/7scenes/'

    scene_names = os.listdir(data_root)
    files = []
    for scene_name in scene_names:
        if not osp.isdir(osp.join(data_root, scene_name)):
            continue
        
        # if 'chess' not in scene_name:
        #     continue
        print(scene_name)

        scene_root = osp.join(data_root, scene_name)
        for seq in os.listdir(scene_root):
            if 'seq' not in seq:
                continue

            base_root = osp.join(data_root, scene_name, seq)
            # if scene_name != 'stairs':
            #     total_imgs = 1000
            # else:
            #     total_imgs = 500

            # for idx in range(0, total_imgs):
            for file_name in os.listdir(base_root):
                if 'frame' not in file_name or 'color.png' not in file_name:
                    continue
                # rgb_path = osp.join(base_root, 'frame-%06d.color.png') %(idx)
                rgb_path = osp.join(base_root, file_name)
                depth_path = rgb_path.replace('.color.png', '.depth.png')
                fx = 585; fy = 585; cx = 320; cy = 240
                cam_in = [fx, fy, cx, cy]
                meta_data = {}
                meta_data['cam_in'] = cam_in
                meta_data['rgb'] = rgb_path.split(split_root)[-1]
                meta_data['depth'] = depth_path.split(split_root)[-1]
                files.append(meta_data)
    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/7scenes/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)
        