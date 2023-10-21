if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/AVD/ActiveVisionDataset'
    split_root = '/mnt/nas/share/home/xugk/data/AVD/'

    scene_names = os.listdir(data_root)
    files = []
    for scene_name in scene_names:
        if scene_name == 'intrinsics':
            continue
        if not osp.isdir(osp.join(data_root, scene_name)):
            continue
        print(scene_name)

        scene_root = osp.join(data_root, scene_name)
        rgb_root = osp.join(scene_root, 'jpg_rgb')
        cam_info_path = osp.join(scene_root.replace('ActiveVisionDataset/', 'ActiveVisionDataset/intrinsics/'), 'cameras.txt')
        for rgb_name in os.listdir(rgb_root):
            rgb_path = osp.join(rgb_root, rgb_name)
            depth_path = rgb_path.replace('jpg_rgb', 'high_res_depth').replace('1.jpg', '3.png')
            with open(cam_info_path, 'r') as f:
                cam_infos = f.readlines()[3].split(' ')
            fx = cam_infos[4]; fy = cam_infos[5]; cx = cam_infos[6]; cy = cam_infos[7]
            cam_in = [float(fx), float(fy), float(cx), float(cy)]
            meta_data = {}
            meta_data['cam_in'] = cam_in
            meta_data['rgb'] = rgb_path.split(split_root)[-1]
            meta_data['depth'] = depth_path.split(split_root)[-1]
            files.append(meta_data)
    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/AVD/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)
        