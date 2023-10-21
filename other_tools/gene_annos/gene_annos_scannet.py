if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random

    import json
    import pickle

    import re
    def microsoft_sorted(input_list):
        return sorted(input_list, key=lambda s: [int(s) if s.isdigit() else s for s in sum(re.findall(r'(\D+)(\d+)', 'a'+s+'0'), ())])


    data_root = '/mnt/nas/share/home/xugk/data/scannet_test/scannet_test'
    split_root = '/mnt/nas/share/home/xugk/data/scannet_test/'

    # data_root = '/mnt/nas/share/home/xugk/data/scannet_test_div_5'
    # split_root = '/mnt/nas/share/home/xugk/data/scannet_test_div_5/'

    scene_names = ['scene0%03d_00' %i for i in range(707, 807) if i != 710]
    # scene_names = ['scene0710_00']
    print('begining...')
    files = []
    for scene_name in scene_names:
        print(scene_name)
        rgb_root = osp.join(data_root, scene_name, 'color')
        rgb_files = microsoft_sorted(os.listdir(rgb_root))
        for i, rgb_file in enumerate(rgb_files):
        # for rgb_file in random.sample(rgb_files, 10):
            # if i >= 100:
            #     continue
            rgb_path = osp.join(rgb_root, rgb_file)
            depth_path = rgb_path.replace('/color/', '/depth/').replace('.jpg', '.png')
            # cam_in = [1165.723022, 1165.738037, 649.094971, 484.765015]
            cam_in_path = osp.join(data_root, scene_name, 'intrinsic/intrinsic_color.txt')
            cam_in_matrix = np.loadtxt(cam_in_path).reshape(4, 4)[:3, :3]
            cam_in = [cam_in_matrix[0][0], cam_in_matrix[1][1], cam_in_matrix[0][2], cam_in_matrix[1][2]]
            meta_data = {}
            meta_data['cam_in'] = cam_in
            meta_data['rgb'] = rgb_path.split(split_root)[-1]
            meta_data['depth'] = depth_path.split(split_root)[-1]
            files.append(meta_data)
    files_dict = dict(files=files)

    # with open('/mnt/nas/share/home/xugk/data/scannet_test/test_annotation_scene0710.json', 'w') as f:
    #     json.dump(files_dict, f)

    with open('/mnt/nas/share/home/xugk/data/scannet_test_div_5/train_annotation_wo_scene0710.json', 'w') as f:
        json.dump(files_dict, f)
        