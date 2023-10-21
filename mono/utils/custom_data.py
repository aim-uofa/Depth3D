import glob
import os
import json
import cv2

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
        }
        datas.append(data_i)
    return datas

def load_data_rgb_depth_intrinsic_norm(rgbs: list, intrinsic, depths=None, depth_scale=None, norms=None):
    if depths is None:
        depths = [None] * len(rgbs)
    else:
        assert depth_scale is not None
        
    if norms is None:
        norms = [None] * len(rgbs)

    if type(intrinsic[0]) is list:
        data = [{'rgb': rgb, 'depth': depths[i], 'depth_scale': depth_scale, 'norm': norms[i], 'intrinsic': intrinsic[i], 'filename': os.path.basename(rgb), 'folder': ''.join(rgb.split('/')[-3:-1])} for i,rgb in enumerate(rgbs)]
    else:
        data = [{'rgb': rgb, 'depth': depths[i], 'depth_scale': depth_scale, 'norm': norms[i], 'intrinsic': intrinsic, 'filename': os.path.basename(rgb), 'folder': ''.join(rgb.split('/')[-3:-1])} for i,rgb in enumerate(rgbs)]
    
    return data

    