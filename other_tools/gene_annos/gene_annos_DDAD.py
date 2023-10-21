if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/dataset/MetricDepth/DDAD/DDAD'
    split_root = '/mnt/nas/share/home/xugk/dataset/MetricDepth/DDAD/'

    file_txt = '/mnt/nas/share/home/gzh/code/NeWCRFs/data_splits/ddad_test_files_with_gt_01.txt'

    with open(file_txt, 'r') as f:
        infos = f.readlines()
    
    files = []
    for i, info in enumerate(infos):
        # if i % 10 == 0:
        #     print(i, '/', len(infos))
        info = info.split(' ')[0]
        pkl_path = info.replace('_rgb.png', '.pkl').replace('/rgb/', '/meta/')
        pkl_path = osp.join('DDAD', pkl_path)
        if not osp.exists(osp.join(split_root, pkl_path)):
            print(pkl_path)
            continue
        meta_data = {}
        meta_data['meta_data'] = pkl_path
        files.append(meta_data)
    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/dataset/MetricDepth/DDAD/test_annotation.json', 'w') as f:
        json.dump(files_dict, f)
        