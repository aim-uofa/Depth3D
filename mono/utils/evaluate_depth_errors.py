import logging
import torch
import numpy as np
logger = logging.getLogger(__name__)


def validate_depth_err(pred, gt, smoothed_criteria, mask=None):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    mask_valid = (gt>0) & (pred>0) if mask is None else (gt>0) & (pred>0)& mask
    
    gt_mask = gt[mask_valid]
    pred_mask = pred[mask_valid]
    
    n_pxl = gt_mask.size
    # invalid evaluation image
    if n_pxl < 10:
        return smoothed_criteria

    # Mean Absolute Relative Error
    rel = np.abs(gt_mask - pred_mask) / gt_mask  # compute errors
    abs_rel_sum = np.sum(rel)
    smoothed_criteria['err_absRel'].AddValue(np.float64(abs_rel_sum), n_pxl)
    
    #Delta 1 Accuracy
    gt_pred = gt_mask / pred_mask
    pred_gt = pred_mask / gt_mask
    gt_pred = np.reshape(gt_pred, (1, -1))
    pred_gt = np.reshape(pred_gt, (1, -1))
    gt_pred_gt = np.concatenate((gt_pred, pred_gt), axis=0)
    ratio_max = np.amax(gt_pred_gt, axis=0)

    delta_1_sum = np.sum(ratio_max < 1.25)
    smoothed_criteria['err_delta1'].AddValue(np.float64(delta_1_sum), n_pxl)
    return smoothed_criteria
