if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/BlendedMVS/BlendedMVS'
    split_root = '/mnt/nas/share/home/xugk/data/BlendedMVS/'

    scene_names = os.listdir(data_root)
    files = []
    for scene_name in scene_names:
        if not osp.isdir(osp.join(data_root, scene_name)):
            continue
        
        # if 'chess' not in scene_name:
        #     continue
        print(scene_name)

        scene_root = osp.join(data_root, scene_name)
        for scene_id in os.listdir(scene_root):
            if 'dataset_full_res_' in scene_root:
                scene_id_root = osp.join(scene_root, scene_id, scene_id, scene_id)
            else:
                scene_id_root = osp.join(scene_root, scene_id)

            rgb_root = osp.join(scene_id_root, 'blended_images')
            for rgb_name in os.listdir(rgb_root):
                if '_masked.jpg' in rgb_name:
                    continue
                rgb_path = osp.join(rgb_root, rgb_name)
                depth_path = rgb_path.replace('blended_images', 'rendered_depth_maps').replace('.jpg', '.pfm')
                cam_path = rgb_path.replace('blended_images', 'cams').replace('.jpg', '_cam.txt')
                with open(cam_path, 'r') as f:
                    infos = f.readlines()

                fx = infos[7].split(' ')[0]
                fy = infos[8].split(' ')[1]
                cx = infos[7].split(' ')[2]
                cy = infos[8].split(' ')[2]

                cam_in = [float(fx), float(fy), float(cx), float(cy)]
                meta_data = {}
                meta_data['cam_in'] = cam_in
                meta_data['rgb'] = rgb_path.split(split_root)[-1]
                meta_data['depth'] = depth_path.split(split_root)[-1]
                files.append(meta_data)
    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/BlendedMVS/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)
        