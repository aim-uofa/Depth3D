if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle
    import pykitti

    # split_file = '/mnt/nas/share/home/jxr/home/code/ZoeDepth/train_test_inputs/kitti_eigen_test_files_with_gt.txt'
    split_file = '/mnt/nas/share/home/jxr/home/code/ZoeDepth/train_test_inputs/kitti_eigen_train_files_with_gt.txt'

    files = []
    with open(split_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(i, '/', len(lines))
            rgb_path, depth_path, _ = line.strip('\n').split(' ')
            if "None" in depth_path:
                continue
            scene_date = rgb_path.split('/')[0]
            scene_drive = rgb_path.split('/')[1].split('_drive_')[-1].split('_sync')[0]

            rgb_path = 'kitti/raw_data/' + rgb_path
            depth_path = 'kitti/depth_annotated/' + depth_path
            anno = {}
            anno['rgb'] = rgb_path
            anno['depth'] = depth_path
            
            pykitti_loader = pykitti.raw('/mnt/nas/datasets/kitti/raw_data', scene_date, scene_drive, frame_range=range(1))
            gt_intrinsic_full = pykitti_loader.calib.K_cam2
            anno['cam_in'] = [gt_intrinsic_full[0][0], gt_intrinsic_full[1][1], gt_intrinsic_full[0][2], gt_intrinsic_full[1][2]]
            files.append(anno)
    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/kitti/train_annotation.json', 'w') as f:
        json.dump(files_dict, f)