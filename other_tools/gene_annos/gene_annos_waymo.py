

if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2

    import json
    import pickle

    data_root = '/mnt/nas/share/home/xugk/data/Waymo/Waymo'
    split_root = '/mnt/nas/share/home/xugk/data/Waymo/'

    files = []
    for phase in ['training', 'testing', 'validation']:
        print('phase :', phase)
        for i, folder in enumerate(['image_0', 'image_1', 'image_2', 'image_3', 'image_4']):
            print(phase, folder)
            folder_root = osp.join(data_root, phase, folder)
            for rgb_file in sorted(os.listdir(folder_root)):
                rgb_path = osp.join(folder_root, rgb_file)
                depth_path = rgb_path.replace('/image_', '/depth_')
                sem_path = rgb_path.replace('/image_', '/semseg_')

                # # check files
                # rgb = cv2.imread(rgb_path)
                # depth = cv2.imread(depth_path, -1)
                # if rgb is None or depth is None:
                #     print('broken files :', rgb_path, depth_path)
                #     continue

                cam_path = osp.join(data_root, phase, 'calib', rgb_file.replace('.png', '.txt'))
                with open(cam_path, 'r') as f:
                    cam_in = f.readlines()[i].split(' ')[1:]
                    cam_in = np.array(cam_in).reshape(3, 4).astype(np.float32)

                meta_data = {}
                meta_data['cam_in'] = [cam_in[0][0], cam_in[1][1], cam_in[0][2], cam_in[1][2]]
                meta_data['rgb'] = rgb_path.split(split_root)[-1]
                meta_data['depth'] = depth_path.split(split_root)[-1]
                meta_data['sem'] = sem_path.split(split_root)[-1]
                meta_data['normal'] = None
                meta_data['ins_plane'] = None

                meta_path = osp.join(data_root, phase, 'meta', folder, rgb_file.replace('.png', '.pkl'))
                os.makedirs(osp.dirname(meta_path), exist_ok=True)
                if not os.path.exists(meta_path): 
                    with open(meta_path, 'wb') as f:
                        pickle.dump(meta_data, f)
                files.append(dict(meta_data=meta_path.split(split_root)[-1]))

    files_dict = dict(files=files)

    with open('/mnt/nas/share/home/xugk/data/Waymo/Waymo/train_annotation_April_all_new.json', 'w') as f:
        json.dump(files_dict, f)
    with open('/mnt/nas/share/home/xugk/data/Waymo/Waymo/train_dataset_info.json', 'w') as f:
        dataset_info = {
            'has_normal': False,
            'has_ins_plane': False,
            'has_semseg': True,
        }
        json.dump(dataset_info, f)
        