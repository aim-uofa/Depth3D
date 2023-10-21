
import json
import torch
import torchvision.transforms as transforms
import os.path
import os.path as osp
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from .__base_dataset__ import BaseDataset
import pickle

class DIODEDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(DIODEDataset, self).__init__(cfg, phase, **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)

        curr_rgb_path = osp.join(self.data_root, meta_data['rgb'])
        curr_depth_path = osp.join(self.depth_root, meta_data['depth'])
        curr_depth_mask_depth = osp.join(self.depth_root, meta_data['depth_mask'])
        curr_sem_path = osp.join(self.sem_root, meta_data['sem']) if ('sem' in meta_data) and (meta_data['sem'] is not None) and self.sem_root is not None else None
        curr_norm_path = osp.join(self.norm_root, meta_data['normal']) if ('normal' in meta_data) and (meta_data['normal'] is not None) and self.norm_root is not None else None

        # load data
        curr_intrinsic = meta_data['cam_in']
        curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path, is_test=False)
        curr_depth = curr_depth.squeeze()
        depth_mask = np.load(curr_depth_mask_depth)
        curr_depth[depth_mask != 1] = 0

        # get semantic labels
        curr_sem = self.load_sem_label(curr_sem_path, H=curr_rgb.shape[0], W=curr_rgb.shape[1])
        # get normal labels
        curr_norm = self.load_norm_label(curr_norm_path, H=curr_rgb.shape[0], W=curr_rgb.shape[1])

        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)

        # get crop size
        transform_paras = dict(random_crop_size = self.random_crop_size)
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
        curr_depth_mask_depth = osp.join(self.depth_root, meta_data['depth_mask'])

        # load data
        curr_intrinsic = meta_data['cam_in']
        curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        curr_depth = curr_depth.squeeze()

        depth_mask = np.load(curr_depth_mask_depth)
        curr_depth[depth_mask != 1] = 0

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
    
    def process_depth(self, depth, rgb):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        return depth