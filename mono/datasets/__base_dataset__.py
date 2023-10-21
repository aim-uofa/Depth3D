import os
import os.path as osp
import json
import torch
import os.path as osp
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
import mono.utils.transform as img_transform
import copy
from mono.utils.comm import get_func
import pickle
import logging
import multiprocessing as mp
import ctypes
from PIL import Image
import re
import h5py
from mono.utils.transform import resize_depth_preserve
'''
Dataset annotations are saved in a Json file. All data, including rgb, depth, and so on, captured within the same frames are saved in teh same dict.
All frames are organized in a list. In each frame, it may contains the some or all of following data format.

# Annotations for the current central RGB/depth cameras.

'rgb':              rgb image in the current frame.
'depth':            depth map in the current frame.
'sem':              semantic mask in the current frame.
'cam_in':           camera intrinsic parameters of the current rgb camera.
'cam_ex':           camera extrinsic parameters of the current rgb camera.
'cam_ex_path':      path to the extrinsic parameters.
'timestamp_rgb':    time stamp of current rgb image.
'''

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

class BaseDataset(Dataset):
    def __init__(self, cfg, phase, **kwargs):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.data_info = kwargs['data_info']

        # root dir for data
        self.data_root = osp.join(self.data_info['database_root'], self.data_info['data_root'])
        # depth/disp data root
        disp_root = self.data_info['disp_root'] if 'disp_root' in self.data_info else None
        self.disp_root = osp.join(self.data_info['database_root'], disp_root) if disp_root is not None else None
        depth_root = self.data_info['depth_root'] if 'depth_root' in self.data_info else None
        self.depth_root = osp.join(self.data_info['database_root'], depth_root) if depth_root is not None \
            else self.data_root
        # meta data root
        meta_data_root = self.data_info['meta_data_root'] if 'meta_data_root' in self.data_info else None
        self.meta_data_root = osp.join(self.data_info['database_root'], meta_data_root) if meta_data_root is not None else None
        # semantic segmentation labels root
        sem_root = self.data_info['semantic_root'] if 'semantic_root' in self.data_info else None
        self.sem_root = osp.join(self.data_info['database_root'], sem_root) if sem_root is not None else None
        # surface normal labels root
        norm_root = self.data_info['norm_root'] if 'norm_root' in self.data_info else None
        self.norm_root = osp.join(self.data_info['database_root'], norm_root) if norm_root is not None else None

        # data annotations path
        self.data_annos_path = osp.join(self.data_info['database_root'], self.data_info['%s_annotations_path' % phase])

        # load annotations
        self.data_info = self.load_annotations()
        whole_data_size = len(self.data_info['files'])

        # sample a subset for training/validation/testing
        # such method is deprecated, each training may get different sample list
        random.seed(100) # set the random seed
        cfg_sample_ratio = cfg.data[phase].sample_ratio
        cfg_sample_size = int(cfg.data[phase].sample_size)
        self.sample_size = int(whole_data_size * cfg_sample_ratio) if cfg_sample_size == -1 \
                           else (cfg_sample_size if cfg_sample_size < whole_data_size else whole_data_size)
        sample_list_of_whole_data = random.sample(list(range(whole_data_size)), self.sample_size)

        self.data_size = self.sample_size
        self.annotations = {'files': [self.data_info['files'][i] for i in sample_list_of_whole_data]}
        self.sample_list = list(range(self.data_size))

        # config transforms for the input and label
        self.transforms_cfg = cfg.data[phase]['pipeline']
        self.transforms_lib = 'mono.utils.transform.'

        self.img_file_type = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        self.np_file_type = ['.npz', '.npy']
        self.pfm_file_type = ['.pfm']
        self.hdf5_file_type = ['.hdf5']
        self.JPG_file_type = ['.JPG']


        # update canonical sparse information
        self.data_basic = copy.deepcopy(kwargs)
        canonical = self.data_basic.pop('canonical_space')
        self.data_basic.update(canonical)
        self.depth_range = kwargs['depth_range'] # predefined depth range for the network
        
        if 'clip_depth_range' in kwargs:
            self.clip_depth_range = kwargs['clip_depth_range'] # predefined depth range for data processing
        else:
            self.clip_depth_range = [1e-6, 1e8]

        self.depth_normalize = kwargs['depth_normalize']

        self.img_transforms = img_transform.Compose(self.build_data_transforms())
        self.EPS = 1e-6

        # dataset info
        self.data_name = cfg.data_name
        self.data_type = cfg.data_type # there are mainly four types, i.e. ['rel', 'sfm', 'stereo', 'lidar']
        self.logger = logging.getLogger()
        self.logger.info(f'{self.data_name} in {self.phase} whole data size: {whole_data_size}') 

        self.random_crop_size = torch.from_numpy(np.array([0,0])) # torch.from_numpy(shared_array)

    def __name__(self):
        return self.data_name

    def __len__(self):
        return self.data_size
    
    def load_annotations(self):
        if not osp.exists(self.data_annos_path):
            raise RuntimeError(f'Cannot find {self.data_annos_path} annotations.')
        
        with open(self.data_annos_path, 'r') as f:
            annos = json.load(f)
        return annos
    
    def build_data_transforms(self):
        transforms_list = []
        for transform in self.transforms_cfg:
            args = copy.deepcopy(transform)
            # insert the canonical space configs
            args.update(self.data_basic)

            obj_name = args.pop('type')
            obj_path = self.transforms_lib + obj_name
            obj_cls = get_func(obj_path)

            obj = obj_cls(**args)
            transforms_list.append(obj)
        return transforms_list

    def load_data(self, path: str, is_rgb_img: bool=False):
        if not osp.exists(path):
            raise RuntimeError(f'{path} does not exist.')

        data_type = osp.splitext(path)[-1]
        if data_type in self.img_file_type:
            if is_rgb_img:
                data = cv2.imread(path)
            else:
                data = cv2.imread(path, -1)
        elif data_type in self.np_file_type:
            data = np.load(path)
        elif data_type in self.pfm_file_type:
            data = readPFM(path).astype(np.float32)
            if len(data.shape) == 2:
                pass
            else:
                data = data[:, :, :-1]
        elif data_type in self.hdf5_file_type:
            assert 'hypersim' in path, "only support .hdf5 file depth of hypersim so far."
            f = h5py.File(path, 'r')
            data = f['dataset'][:]
        elif data_type in self.JPG_file_type:
            assert 'ETH3D' in path, "only support .JPG file depth of ETH3D so far."
            if is_rgb_img:
                data = cv2.imread(path)
            else:
                f = open(path, 'r')
                data = np.fromfile(f, np.float32)
                data = data.reshape((4032, 6048))
        else:
            raise RuntimeError(f'{data_type} is not supported in current version.')
        
        return data
    
    def __getitem__(self, idx: int) -> dict:
        # protect the mistakes of data loading.
        if self.phase == 'test':
            return self.get_data_for_test(idx)
        else:
            while True:
                try:
                    return self.get_data_for_trainval(idx)
                    break
                except:
                    self.logger.info('data __getitem__ error!!! idx = idx - 1')
                    if idx > 0:
                        idx = idx - 1
                    else:
                        idx = self.__len__() - 1
                    continue

        # NOTE: if you don't want the data loading protection, use the code below instead.
        """ 
        if self.phase == 'test':
            return self.get_data_for_test(idx)
        else:
            return self.get_data_for_trainval(idx)
        """
    
    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)

        curr_rgb_path = osp.join(self.data_root, meta_data['rgb'])
        curr_depth_path = osp.join(self.depth_root, meta_data['depth'])
        curr_sem_path = osp.join(self.sem_root, meta_data['sem']) if ('sem' in meta_data) and (meta_data['sem'] is not None) and self.sem_root is not None else None
        curr_norm_path = osp.join(self.norm_root, meta_data['normal']) if ('normal' in meta_data) and (meta_data['normal'] is not None) and self.norm_root is not None else None

        # load data
        curr_intrinsic = meta_data['cam_in']
        curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path, is_test=False)
        # get semantic labels
        curr_sem = self.load_sem_label(curr_sem_path, H=curr_rgb.shape[0], W=curr_rgb.shape[1])
        # get normal labels
        curr_norm = self.load_norm_label(curr_norm_path, H=curr_rgb.shape[0], W=curr_rgb.shape[1])

        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)

        # get crop size
        transform_paras = dict(random_crop_size = self.random_crop_size) # change the crop size of RandomCrop transformation.
        rgbs, depths, intrinsics, cam_models, norms, other_labels, transform_paras = self.img_transforms(
            images=[curr_rgb, ],
            labels=[curr_depth, ],
            intrinsics=[curr_intrinsic, ],
            cam_models=[curr_cam_model, ],
            norms=[curr_norm, ],
            other_labels=[curr_sem, ],
            transform_paras=transform_paras,
        )

        # process sky masks
        sem_mask = other_labels[0].int()
        mask_depth_valid = depths[0] > 1e-8
        invalid_sky_region = (sem_mask==142) & (mask_depth_valid)
        if self.data_type in ['lidar', 'sfm']:
            sem_mask[invalid_sky_region] = -1
        
        # clip depth map
        depth_out = self.normalize_depth(depths[0])
        # set the depth in sky region to the maximum depth
        depth_out[sem_mask==142] = -1
        filename = os.path.basename(meta_data['rgb'])
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
        ]
        pad = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
        data = dict(
            input=rgbs[0],
            target=depth_out,
            intrinsic=curr_intrinsic_mat,
            filename=filename,
            dataset=self.data_name,
            cam_model=cam_models_stacks,
            pad=torch.tensor(pad),
            data_type=[self.data_type, ],
            norms=norms,
            sem_mask=sem_mask.int(),
            transformed_rgb_shape=(rgbs[0].shape[1], rgbs[0].shape[2]),
        )
        return data

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        curr_rgb_path = osp.join(self.data_root, meta_data['rgb'])
        curr_depth_path = osp.join(self.depth_root, meta_data['depth'])

        # load data
        curr_intrinsic = meta_data['cam_in']
        curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path, is_test=True)

        # resize to (504, 756) while testing ETH3D dataset
        data_type = osp.splitext(curr_rgb_path)[-1]
        if data_type in self.JPG_file_type:
            curr_rgb = cv2.resize(curr_rgb, (756, 504), cv2.INTER_LINEAR)
            curr_depth = resize_depth_preserve(curr_depth, (504, 756)).astype(np.float32)
            curr_intrinsic = [curr_intrinsic[0] / 8, curr_intrinsic[1] / 8, curr_intrinsic[2] / 8, curr_intrinsic[3] / 8]
            self.logger.info('resize ETH image to (504, 756) for fast testing.')

        ori_h, ori_w, _ = curr_rgb.shape

        # pseudo smantic labels
        curr_sem = self.load_sem_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1])
        # pseudo normal labels
        curr_norm = self.load_norm_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1])

        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)

        # get crop size
        transform_paras = dict()
        rgbs, _, intrinsics, cam_models, norms, other_labels, transform_paras = self.img_transforms(
            images=[curr_rgb, ],
            labels=[curr_depth, ],
            intrinsics=[curr_intrinsic, ],
            cam_models=[curr_cam_model, ],
            norms=[curr_norm, ],
            other_labels=[curr_sem, ],
            transform_paras=transform_paras,
        )

        # depth in original size and original metric
        curr_depth = self.clip_depth(curr_depth) * self.depth_range[1]

        filename = osp.basename(meta_data['rgb'])
        filepath = meta_data['rgb']
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0, 0, 0, 0]
        scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
        ]
        raw_rgb = torch.from_numpy(curr_rgb)
        raw_depth = torch.from_numpy(curr_depth)

        data = dict(
            input=rgbs[0],
            # target=depth_out,
            intrinsic=curr_intrinsic_mat,
            filename=filename,
            dataset=self.data_name,
            cam_model=cam_models_stacks,
            pad=pad,
            scale=scale_ratio,
            raw_rgb=raw_rgb,
            raw_depth=raw_depth,
            raw_intrinsic=curr_intrinsic,
            filepath=filepath,
            depth_path=curr_depth_path,
        )
        return data

    def clip_depth(self, depth: np.array) -> np.array:
        depth[(depth>self.clip_depth_range[1] | (depth<self.clip_depth_range[0]))] = -1
        depth /= self.depth_range[1]
        depth[depth<self.EPS] = -1
        return depth

    def normalize_depth(self, depth: np.array) -> np.array:
        depth /= self.depth_range[1]
        depth[depth<self.EPS] = -1
        return depth
    
    def process_depth(self, depth: np.array, rgb: np.array=None) -> np.array:
        return depth

    def create_cam_model(self, H: int, W: int, intrinsics: list) -> np.array:
        '''
        Encode the camera model (focal length and principle point) to a 4-channel map.
        '''
        fx, fy, u0, v0 = intrinsics
        f = (fx + fy) / 2.0
        # principle point location
        x_row = np.arange(0, W).astype(np.float32)
        x_row_center_norm = (x_row - u0) / W
        x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

        y_col = np.arange(0, H).astype(np.float32)
        y_col_center_norm = (y_col - v0) / H
        y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

        # FoV
        fov_x = np.arctan(x_center / (f / W))
        fov_y = np.arctan(y_center / (f / H))

        cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
        return cam_model
    
    def check_data(self, data_dict: dict):
        for k, v in data_dict.items():
            if v is None:
                self.logger.info(f'{self.data_name}, {k} cannot be read!')

    def intrinsics_list2mat(self, intrinsics: torch.tensor) -> torch.tensor:
        intrinsics_mat = torch.zeros((3,3)).float()
        intrinsics_mat[0, 0] = intrinsics[0]
        intrinsics_mat[1, 1] = intrinsics[1]
        intrinsics_mat[0, 2] = intrinsics[2]
        intrinsics_mat[1, 2] = intrinsics[3]
        intrinsics_mat[2, 2] = 1.0
        return intrinsics_mat
    
    def load_meta_data(self, anno: dict) -> dict:
        '''
        Load meta data information.
        '''
        if self.meta_data_root is not None and 'meta_data' in anno:
            meta_data_path = osp.join(self.meta_data_root, anno['meta_data'])
            with open(meta_data_path, 'rb') as f:
                meta_data = pickle.load(f)
            meta_data.update(anno)
        else:
            meta_data = anno
        return meta_data
    
    def load_rgb_depth(self, rgb_path: str, depth_path: str, is_test=False) -> (np.array, np.array):
        '''
        Load the rgb and depth map with the paths.
        '''
        rgb = self.load_data(rgb_path, is_rgb_img=True)
        if rgb is None:
            assert False
        else:
            if depth_path is not None:
                depth = self.load_data(depth_path)
            else:
                assert False
        
        # if training or validation, resize the depth map to the RGB size
        if not is_test:
            depth = resize_depth_preserve(depth, (rgb.shape[0], rgb.shape[1]))

        self.check_data(
            dict(
                rgb_path=rgb,
                depth_path=depth,
            )
        )
        
        depth = depth.astype(np.float32)
        depth = self.process_depth(depth, rgb)
        return rgb, depth
    
    def load_sem_label(self, sem_path, H, W, sky_id=142):
        sem_label = cv2.imread(sem_path, 0) if sem_path is not None \
            else np.ones((H, W), dtype=np.int32) * -1
        if sem_label is None:
            sem_label = np.ones((H, W), dtype=np.int32) * -1
        # set dtype to int before
        sem_label = sem_label.astype(np.int32)
        sem_label[sem_label==255] = -1
        return sem_label
    
    def load_norm_label(self, norm_path, H, W):
        if norm_path is not None:
            norm_gt = Image.open(norm_path).convert("RGB").resize(size=(W, H), resample=Image.NEAREST)
            norm_gt = np.array(norm_gt).astype(np.uint8)
            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
        else:
            norm_gt = np.zeros((H, W, 3)).astype(np.float32)
        return norm_gt
    
    def set_random_crop_size(self, random_crop_size):
        self.random_crop_size[0] = random_crop_size[0]
        self.random_crop_size[1] = random_crop_size[1]

