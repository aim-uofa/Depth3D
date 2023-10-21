if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle
    import pykitti

    # split_file = '/mnt/nas/share/home/xugk/data/dataset_split_txt/nyudepthv2_test_files_with_gt.txt'
    split_file = '/mnt/nas/share/home/xugk/data/dataset_split_txt/nyudepthv2_train_files_with_gt.txt'

    files = []
    with open(split_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(i, '/', len(lines))
            rgb_path, depth_path, _ = line.strip('\n').split(' ')

            rgb_path = 'nyu/' + rgb_path
            depth_path = 'nyu/' + depth_path
            anno = {}
            anno['rgb'] = rgb_path
            anno['depth'] = depth_path
            anno['cam_in'] = [518.8579, 519.46961, 325.58245, 253.73617]
            files.append(anno)
    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/nyu/train_annotation.json', 'w') as f:
        json.dump(files_dict, f)