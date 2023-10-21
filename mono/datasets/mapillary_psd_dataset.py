# NOTE: gt depth is achieved from sfm, better need to constraint the employment of loss.
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from .__base_dataset__ import BaseDataset
import pickle

class MapillaryPSDDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(MapillaryPSDDataset, self).__init__(cfg, phase, **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def process_depth(self, depth, rgb):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        h, w, _ = rgb.shape
        depth_resize = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth_resize