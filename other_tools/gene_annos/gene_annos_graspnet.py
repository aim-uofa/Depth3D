if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/graspnet'
    split_root = '/mnt/nas/share/home/xugk/data/graspnet/'

    files = []
    for scene_name in os.listdir(data_root):
        print(scene_name)
        if scene_name == 'annotations':
            continue
        scene_root = osp.join(data_root, scene_name)

        for scene_id in os.listdir(scene_root):
            scene_id_root = osp.join(scene_root, scene_id)

            for camera_type in ["kinect", "realsense"]:
                camera_type_root = osp.join(scene_id_root, camera_type)
                K = np.load(osp.join(camera_type_root, 'camK.npy'))
                fx = K[0][0]; fy = K[1][1]; cx = K[0][2]; cy = K[1][2]
                rgb_root = osp.join(camera_type_root, 'rgb')
                for file_name in os.listdir(rgb_root):
                    rgb_path = osp.join(rgb_root, file_name)
                    depth_path = rgb_path.replace('/rgb/', '/depth/')

                    cam_in = [fx, fy, cx, cy]
                    meta_data = {}
                    meta_data['cam_in'] = cam_in
                    meta_data['rgb'] = rgb_path.split(split_root)[-1]
                    meta_data['depth'] = depth_path.split(split_root)[-1]
                    files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/graspnet/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)