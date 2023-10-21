import os
import os.path as osp
import json
from mono.utils.database import load_data_info
import pickle

data_info = {}
load_data_info('data_info', data_info=data_info)

# print(data_info)

def load_json(path):
    with open(path, 'r') as f:
        annos = json.load(f)
    return annos

def check(annos, data_root, depth_root, meta_data_root, semantic_root, database_root):
    for an in annos['files']:
        if meta_data_root is not None and 'meta_data' in an:
            meta_data_path = osp.join(database_root, meta_data_root, an['meta_data'])
            if not osp.exists(meta_data_path):
                print(f'{meta_data_path} does not exists')
            with open(meta_data_path, 'rb') as f:
                meta = pickle.load(f)
            an.update(meta)
        else:
            an = meta

        rgb_path = osp.join(database_root, depth_root, an['depth'])
        if 'depth' in an:
            if an['depth'] is None:
                continue
            depth_path = osp.join(database_root, depth_root, an['depth'])
        elif 'disp' in an:
            depth_path = osp.join(database_root, depth_root, an['disp'])
        else:
            depth_path = None
        
        if 'sem' in an and an['sem'] is not None:
            sem_path = osp.join(database_root, semantic_root, an['sem'])
            if not osp.exists(sem_path):
                print(f'{sem_path} does not exists')
        
        if 'ins_planes' in an and an['ins_planes'] is not None:
            ins_path = osp.join(database_root, data_root, an['ins_planes'])
            if not osp.exists(ins_path):
                print(f'{ins_path} does not exists')
        
        if 'normal' in an and an['normal'] is not None:
            normal_path = osp.join(database_root, data_root, an['normal'])
            if not osp.exists(normal_path):
                print(f'{normal_path} does not exists')
        
        if 'point_info' in an and an['point_info'] is not None:
            point_info_path = osp.join(database_root, data_root, an['point_info'])
            if not osp.exists(point_info_path):
                print(f'{point_info_path} does not exists')
        
        if not osp.exists(rgb_path):
            print(f'{rgb_path} does not exists')
        if depth_path and not osp.exists(depth_path):
            print(f'{depth_path} does not exists')

def loop_datasets():
    ignore_list = []
    for data_name, files, in data_info.items():
        if data_name in ignore_list:
            continue
        print('data_name :', data_name)

        depth_root = files['data_root']

        if 'meta_data_root' in files:
            meta_data_root = files['meta_data_root']
        else:
            meta_data_root = None
        
        if 'semantic_root' in files:
            semantic_root = files['semantic_root']
        else:
            semantic_root = None

        # try:
        #     train_path = osp.join(files['database_root'], files['train_annotations_path'])
        #     trains = load_json(train_path)
        #     print('Train size :', len(trains['files']))
        #     check(trains, files['data_root'], depth_root, meta_data_root, semantic_root, database_root=files['database_root'])
        # except KeyboardInterrupt:
        #     raise KeyboardInterrupt
        # except:
        #     print('Missing train annotaitons.')

        try:
            val_path = osp.join(files['database_root'], files['val_annotations_path'])
            vals = load_json(val_path)
            print('Val size :', len(vals['files']))
            check(vals, files['data_root'], depth_root, meta_data_root, semantic_root, database_root=files['database_root'])
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print('Missing val annotaitons.')

        try:
            test_path = osp.join(files['database_root'], files['test_annotations_path'])
            tests = load_json(test_path)
            print('Test size :', len(tests['files']))
            check(tests, files['data_root'], depth_root, meta_data_root, semantic_root, database_root=files['database_root'])
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print('Missing test annotaitons.')
        

if __name__ == '__main__':
    loop_datasets()