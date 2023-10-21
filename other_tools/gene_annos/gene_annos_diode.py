if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/datasets2/MetricDepth/diode/diode'
    split_root = '/mnt/nas/share/home/xugk/datasets2/MetricDepth/diode/'

    files = []
    for scene_type in ['indoors', 'outdoor']:
        type_root = osp.join(data_root, 'train', scene_type)
        for scene_name in os.listdir(type_root):
            for scan in os.listdir(osp.join(type_root, scene_name)):
                scan_root = osp.join(type_root, scene_name, scan)
                if not osp.isdir(scan_root):
                    continue
                for rgb_path in os.listdir(scan_root):
                    if not rgb_path.endswith('.png'):
                        continue
                    rgb_path = osp.join(scan_root, rgb_path)
                    depth_path = rgb_path.replace('.png', '_depth.npy')
                    depth_mask_path = rgb_path.replace('.png', '_depth_mask.npy')
                    sem_path = rgb_path.replace('.png', '_mask.png').replace('/diode/diode/', '/diode/diode_mask/')

                    cam_in = [886.81, 927.06, 512, 384]
                    meta_data = {}
                    meta_data['cam_in'] = cam_in
                    meta_data['rgb'] = rgb_path.split(split_root)[-1]
                    meta_data['depth'] = depth_path.split(split_root)[-1]
                    meta_data['depth_mask'] = depth_mask_path.split(split_root)[-1]
                    if scene_type == 'outdoor':
                        meta_data['sem'] = sem_path.split(split_root)[-1]
                    files.append(meta_data)

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/datasets2/MetricDepth/diode/train_annotations.json', 'w') as f:
        json.dump(files_dict, f)