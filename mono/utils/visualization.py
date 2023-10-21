import matplotlib.pyplot as plt
import os, cv2
import numpy as np
from mono.utils.transform import gray_to_colormap
import shutil
import glob
from mono.utils.running import main_process
import torch
from html4vision import Col, imagetable

# kappa to exp error (only applicable to AngMF distribution)
def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha


# normal vector to rgb values
def norm_to_rgb(norm):
    # norm: (B, H, W, 3)
    norm_rgb = ((norm[0, ...] + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)
    return norm_rgb

def save_raw_imgs( 
    pred: torch.tensor,  
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str,
    scale: float=200.0, 
    target: torch.tensor=None,
    ):
    """
    Save raw GT, predictions, RGB in the same file.
    """
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_d.png'), (pred*scale).astype(np.uint16))
    if target is not None:
        cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_gt.png'), (target*scale).astype(np.uint16))
    

def save_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    target: torch.tensor, 
    normal,
    normal_kappa,
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str,
    model_type: str='convnext', 
    tb_logger=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, target_scale, pred_color, target_color, normal_color, normal_angular_error_color = get_data_for_log(pred, target, rgb, normal, normal_kappa, model_type)
    rgb = rgb.transpose((1, 2, 0))
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_pred.png'), pred_scale, cmap='rainbow')
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_gt.png'), target_scale, cmap='rainbow')
    if normal_color is not None and normal_angular_error_color is not None:
        cat_img = np.concatenate([rgb, pred_color, target_color, normal_color, normal_angular_error_color], axis=0)
    else:
        cat_img = np.concatenate([rgb, pred_color, target_color], axis=0)
    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        # tb_logger.add_image(f'{filename[:-4]}_rgb.jpg', rgb, iter)
        # tb_logger.add_image(f'{filename[:-4]}_pred.jpg', gray_to_colormap(pred_scale).transpose((2, 0, 1)), iter)
        # tb_logger.add_image(f'{filename[:-4]}_gt.jpg', gray_to_colormap(target_scale).transpose((2, 0, 1)), iter)
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)

def get_data_for_log(pred: torch.tensor, target: torch.tensor, rgb: torch.tensor, normal, normal_kappa, model_type):
    if 'convnext' in model_type:
        mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
        std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]
    elif ('beit' in model_type) or ('swinv2' in model_type):
        mean = np.array([127.5, 127.5, 127.5])[:, np.newaxis, np.newaxis]
        std= np.array([127.5, 127.5, 127.5])[:, np.newaxis, np.newaxis]
    else:
        raise ValueError

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()

    pred[pred<0] = 0
    target[target<0] = 0
    max_scale = max(pred.max(), target.max())
    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    target_scale = (target/max_scale * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred)
    target_color = gray_to_colormap(target)
    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))
    target_color = cv2.resize(target_color, (rgb.shape[2], rgb.shape[1]))

    if normal is not None:
        normal = normal.cpu().permute(0, 2, 3, 1).numpy()
        normal_color = norm_to_rgb(normal)
    else:
        normal_color = None
    
    if normal_kappa is not None:
        normal_kappa = normal_kappa.cpu().permute(0, 2, 3, 1).numpy()
        normal_angular_error_color = kappa_to_alpha(normal_kappa)
        normal_angular_error_color[normal_angular_error_color > 60] = 60
        normal_angular_error_color[normal_angular_error_color < 0] = 0
        normal_angular_error_color = gray_to_colormap(np.squeeze(normal_angular_error_color))
    else:
        normal_angular_error_color = None

    rgb = ((rgb * std) + mean).astype(np.uint8)
    return rgb, pred_scale, target_scale, pred_color, target_color, normal_color, normal_angular_error_color


def create_html(name2path, save_path='index.html', size=(256, 384)):
    # table description
    cols = []
    for k, v in name2path.items():
        col_i =  Col('img', k, v) # specify image content for column
        cols.append(col_i)
    # html table generation
    imagetable(cols, out_file=save_path, imsize=size)


def visual_train_data(gt_depth, rgb, filename, wkdir, replace=False):
    gt_depth = gt_depth.cpu().squeeze().numpy()
    rgb = rgb.cpu().squeeze().numpy()

    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]
    mask = gt_depth > 0
    
    rgb = ((rgb * std) + mean).astype(np.uint8).transpose((1, 2, 0))
    gt_vis = gray_to_colormap(gt_depth)
    if replace:
        rgb[mask] = gt_vis[mask]
    merge = np.concatenate([rgb, gt_vis], axis=0)
    
    save_path = os.path.join(wkdir, 'test_train', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, merge)


def create_dir_for_validate_meta(work_dir, iter_id):
    curr_folders = glob.glob(work_dir + '/online_val/*0')
    curr_folders = [i for i in curr_folders if os.path.isdir(i)]
    if len(curr_folders) > 8:
        curr_folders.sort()
        del_foler = curr_folders.pop(0)
        print(del_foler)
        if main_process():
            # only rank==0 do it
            shutil.rmtree(del_foler)
            os.remove(del_foler + '.html')
    
    save_val_meta_data_dir = os.path.join(work_dir, 'online_val', '%08d'%iter_id)
    os.makedirs(save_val_meta_data_dir, exist_ok=True)
    return save_val_meta_data_dir


def vis_surface_normal(normal: torch.tensor, mask: torch.tensor=None) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.cpu().numpy().squeeze()
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    if mask is not None:
        mask = mask.cpu().numpy().squeeze()
        normal_vis[~mask] = 0
    return normal_vis