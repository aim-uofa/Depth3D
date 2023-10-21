if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/datasets2/MetricDepth/tartanair/tartanair'
    split_root = '/mnt/nas/share/home/xugk/datasets2/MetricDepth/tartanair/'


    files = []

    for scene_name in os.listdir(data_root):
        print(scene_name)
        scene_root = osp.join(data_root, scene_name, scene_name)
        if not (osp.isdir(scene_root) and osp.exists(scene_root)):
            continue
        
        for folder in ['Easy', 'Hard']:
            folder_root = osp.join(scene_root, folder)
            if not osp.exists(folder_root):
                continue
            
            for subfolder in os.listdir(folder_root):
                subfolder_root = osp.join(folder_root, subfolder)
            
                for rgb_folder in ['image_left', 'image_right']:
                    rgb_root = osp.join(subfolder_root, rgb_folder)
                    if not (osp.isdir(rgb_root) and osp.exists(rgb_root)):
                        continue
                    
                    for img_name in os.listdir(rgb_root):
                        cam_in = [320.0, 320.0, 320.0, 240.0]
                        img_path = osp.join(rgb_root, img_name)
                        depth_path = img_path.replace('/image_', '/depth_').replace('.png', '_depth.npy')
                        sem_path = img_path.replace('/tartanair/tartanair/', '/tartanair/tartanair_mask/').replace('.png', '_mask.png')

                        meta_data = {}
                        meta_data['cam_in'] = cam_in
                        meta_data['rgb'] = img_path.split(split_root)[-1]
                        meta_data['depth'] = depth_path.split(split_root)[-1]
                        meta_data['sem'] = sem_path.split(split_root)[-1]
                        files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/datasets2/MetricDepth/tartanair/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)