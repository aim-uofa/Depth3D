# if __name__=='__main__':
#     import os
#     import os.path as osp
#     import numpy as np
#     import cv2
#     import random

#     import json
#     import pickle
#     from random import choice

#     data_root = '/mnt/nas/datasets/nuscenes/nuscenes'
#     split_root = '/mnt/nas/datasets/nuscenes/'


#     files = []
#     cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

#     for i in range(1000):
#         cam_name = choice(cam_names)

#         rgb_root = osp.join(data_root, 'samples/%s' %cam_name)

#         rgb_paths = os.listdir(rgb_root)
#         rgb_path = choice(rgb_paths)

#         rgb_path = osp.join(rgb_root, rgb_path)
#         depth_path = rgb_path.replace('/CAM_', '/DEP_').replace('.jpg', '.png')
#         print(depth_path)
#         if not osp.exists(depth_path):
#             continue

#         # intrinsic for front camera
#         fx = 1266.417203046554; fy = 1266.417203046554; cx = 816.2670197447984; cy = 491.50706579294757

#         cam_in = [fx, fy, cx, cy]
#         meta_data = {}
#         meta_data['cam_in'] = cam_in
#         meta_data['rgb'] = rgb_path.split(split_root)[-1]
#         meta_data['depth'] = depth_path.split(split_root)[-1]
#         files.append(meta_data)

#     files_dict = dict(files=files)

#     with open('/mnt/nas/share/home/xugk/data/nuscenes/test_annotation.json', 'w') as f:
#         json.dump(files_dict, f)
        
if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import random
    import mmcv
    import pickle
    import json
    from random import choice

    data_root = '/mnt/nas/datasets/nuscenes/nuscenes'
    split_root = '/mnt/nas/datasets/nuscenes/'

    infos = mmcv.load('/mnt/nas/share/home/xugk/data/nuscenes/nuscenes/nuscenes_infos_test_bevdepth.pkl')
    # cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    cam_names = ['CAM_FRONT']
    files = []

    ids = random.sample(range(len(infos)), 1000)
    for i in ids:
        cam_name = choice(cam_names)
        rgb_path = infos[i]['cam_infos'][cam_name]['filename']
        rgb_path = osp.join(data_root, rgb_path)

        depth_path = rgb_path.replace('/CAM_', '/DEP_').replace('.jpg', '.png')
        intrinsic = infos[i]['cam_infos'][cam_name]['calibrated_sensor']['camera_intrinsic']
        fx = intrinsic[0][0]; fy = intrinsic[1][1]; cx = intrinsic[0][2]; cy = intrinsic[1][2]
        print(depth_path)
        if not osp.exists(depth_path):
            print('passing :', depth_path)
            continue

        cam_in = [fx, fy, cx, cy]
        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path.split(split_root)[-1]
        meta_data['depth'] = depth_path.split(split_root)[-1]
        files.append(meta_data)

    files_dict = dict(files=files)
    
    with open('/mnt/nas/share/home/xugk/data/nuscenes/test_annotation.json', 'w') as f:
        json.dump(files_dict, f)