if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/hypersim'
    split_root = '/mnt/nas/share/home/xugk/data/hypersim/'

    files = []
    for scene_name in os.listdir(data_root):
        print(scene_name)
        scene_root = osp.join(data_root, scene_name, 'images')
        if not (osp.isdir(scene_root) and osp.exists(scene_root)):
            continue
        
        for folder in os.listdir(scene_root):
            if '_final_preview' not in folder:
                continue
            folder_root = osp.join(scene_root, folder)
            for file_name in os.listdir(folder_root):
                rgb_path = osp.join(folder_root, file_name)
                depth_path = os.path.splitext(os.path.splitext(rgb_path)[0])[0].replace('_final_preview', '_geometry_hdf5') + '.depth_meters.hdf5'
                if not osp.exists(depth_path):
                    assert False

                height_pixels = 768; width_pixels = 1024
                fov_x = np.pi/3.0
                fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x/2.0) / width_pixels)

                fx = (width_pixels / 2) / np.tan(fov_x / 2) ; fy = (height_pixels / 2) / np.tan(fov_y / 2)
                cx = width_pixels / 2; cy = height_pixels / 2

                cam_in = [fx, fy, cx, cy]
                meta_data = {}
                meta_data['cam_in'] = cam_in
                meta_data['rgb'] = rgb_path.split(split_root)[-1]
                meta_data['depth'] = depth_path.split(split_root)[-1]
                files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/hypersim/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)