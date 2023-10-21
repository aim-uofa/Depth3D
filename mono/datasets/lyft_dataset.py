
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

class LyftDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(LyftDataset, self).__init__(cfg, phase, **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def process_depth(self, depth, rgb):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        return depth