

import os
import os.path as osp
import re
import ast
import sys
CODE_SPACE=osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
# sys.path.append(CODE_SPACE)
# os.chmod(CODE_SPACE)
from tabulate import tabulate

def check_log_completeness(log_path):
    try:
        with open(log_path, 'r') as f:
            log_last_line = f.readlines()[-1]
            if 'Evaluation finished.' in log_last_line:
                return True
            else:
                return False
    except:
        return False

def read_evaluation_from_log(log_path):
    with open(log_path, 'r') as f:
        metric_info, median_info, global_info = f.readlines()[-7:-1:2]
    
    def search_dict(info):
        info = info.strip('\n')
        pattern = r"{.*?}"
        match = re.search(pattern, info)
        if match:
            dict_str = match.group()
            dictionary = eval(dict_str)
            return dictionary
        else:
            return None

    metric_info = search_dict(metric_info)
    median_info = search_dict(median_info)
    global_info = search_dict(global_info)

    info = {
        "metric": metric_info,
        "median": median_info,
        "global": global_info,   
    }

    return info
    

if __name__ == "__main__":

    log_root = osp.join(CODE_SPACE, 'show_dirs', 'logs_of_technical_report')

    cmd_list = [
        'bash scripts/test/beit/test_beit_kitti.sh', # 0, beit, kitti
        'bash scripts/test/beit/test_beit_nyu.sh', # 1, beit, nyu
        'bash scripts/test/beit/test_beit_scannet.sh', # 2, beit, scannet
        'bash scripts/test/beit/test_beit_diode.sh', # 3, beit, diode
        'bash scripts/test/beit/test_beit_eth3d.sh', # 4, beit, eth3d

        'bash scripts/test/convnext/test_convnext_kitti.sh', # 5, convnext, kitti
        'bash scripts/test/convnext/test_convnext_nyu.sh', # 6, convnext, nyu
        'bash scripts/test/convnext/test_convnext_scannet.sh', # 7, convnext, scannet
        'bash scripts/test/convnext/test_convnext_diode.sh', # 8, convnext, diode
        'bash scripts/test/convnext/test_convnext_eth3d.sh', # 9, convnext, eth3d

        'bash scripts/test/swin2/test_swin2_kitti.sh', # 10, swin2, kitti
        'bash scripts/test/swin2/test_swin2_nyu.sh', # 11, swin2, nyu
        'bash scripts/test/swin2/test_swin2_scannet.sh', # 12, swin2, scannet
        'bash scripts/test/swin2/test_swin2_diode.sh', # 13, swin2, diode
        'bash scripts/test/swin2/test_swin2_eth3d.sh', # 14, swin2, eth3d
    ]

    log_path_list = [
        osp.join(log_root, 'test', 'log_test_beit_kitti.txt'),# 0, beit, kitti
        osp.join(log_root, 'test', 'log_test_beit_nyu.txt'), # 1, beit, nyu
        osp.join(log_root, 'test', 'log_test_beit_scannet.txt'), # 2, beit, scannet
        osp.join(log_root, 'test', 'log_test_beit_diode.txt'), # 3, beit, diode
        osp.join(log_root, 'test', 'log_test_beit_eth3d.txt'), # 4, beit, eth3d

        osp.join(log_root, 'test', 'log_test_convnext_kitti.txt'),# 5, convnext, kitti
        osp.join(log_root, 'test', 'log_test_convnext_nyu.txt'), # 6, convnext, nyu
        osp.join(log_root, 'test', 'log_test_convnext_scannet.txt'), # 7, convnext, scannet
        osp.join(log_root, 'test', 'log_test_convnext_diode.txt'), # 8, convnext, diode
        osp.join(log_root, 'test', 'log_test_convnext_eth3d.txt'), # 9, convnext, eth3d

        osp.join(log_root, 'test', 'log_test_swin2_kitti.txt'),# 10, swin2, kitti
        osp.join(log_root, 'test', 'log_test_swin2_nyu.txt'), # 11, swin2, nyu
        osp.join(log_root, 'test', 'log_test_swin2_scannet.txt'), # 12, swin2, scannet
        osp.join(log_root, 'test', 'log_test_swin2_diode.txt'), # 13, swin2, diode
        osp.join(log_root, 'test', 'log_test_swin2_eth3d.txt'), # 14, swin2, eth3d
    ]

    assert len(cmd_list) == len(log_path_list)
    cmd_num = len(cmd_list)
    print('Starting evaluation, it will take several hours of time.')

    for i, (cmd, log_path) in enumerate(zip(cmd_list, log_path_list)):
        print('Evaluating and saving log files:', i + 1, '/', cmd_num)
        if not (osp.exists(log_path) and check_log_completeness(log_path)):
            os.makedirs(osp.dirname(log_path), exist_ok=True)
            cmd_save = cmd + " > " + log_path
            print(cmd_save)
            os.system(cmd_save)
        else:
            print('file %s exists, continue...' %log_path)
        print('\n')
    
    row_cnt = 0

    # beit
    beit_kitti_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['median']['abs_rel']
    beit_nyu_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['median']['abs_rel']
    beit_scannet_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['median']['abs_rel']
    beit_diode_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['median']['abs_rel']
    beit_eth3d_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['median']['abs_rel']

    beit_kitti_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['median']['delta1']
    beit_nyu_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['median']['delta1']
    beit_scannet_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['median']['delta1']
    beit_diode_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['median']['delta1']
    beit_eth3d_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['median']['delta1']
    
    beit_kitti_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['global']['abs_rel']
    beit_nyu_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['global']['abs_rel']
    beit_scannet_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['global']['abs_rel']
    beit_diode_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['global']['abs_rel']
    beit_eth3d_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['global']['abs_rel']

    beit_kitti_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['global']['delta1']
    beit_nyu_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['global']['delta1']
    beit_scannet_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['global']['delta1']
    beit_diode_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['global']['delta1']
    beit_eth3d_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['global']['delta1']
    row_cnt +=1


    # convnext
    convnext_kitti_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['median']['abs_rel']
    convnext_nyu_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['median']['abs_rel']
    convnext_scannet_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['median']['abs_rel']
    convnext_diode_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['median']['abs_rel']
    convnext_eth3d_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['median']['abs_rel']

    convnext_kitti_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['median']['delta1']
    convnext_nyu_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['median']['delta1']
    convnext_scannet_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['median']['delta1']
    convnext_diode_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['median']['delta1']
    convnext_eth3d_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['median']['delta1']

    convnext_kitti_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['global']['abs_rel']
    convnext_nyu_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['global']['abs_rel']
    convnext_scannet_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['global']['abs_rel']
    convnext_diode_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['global']['abs_rel']
    convnext_eth3d_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['global']['abs_rel']

    convnext_kitti_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['global']['delta1']
    convnext_nyu_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['global']['delta1']
    convnext_scannet_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['global']['delta1']
    convnext_diode_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['global']['delta1']
    convnext_eth3d_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['global']['delta1']
    row_cnt +=1


    # swin2
    swin2_kitti_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['median']['abs_rel']
    swin2_nyu_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['median']['abs_rel']
    swin2_scannet_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['median']['abs_rel']
    swin2_diode_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['median']['abs_rel']
    swin2_eth3d_median_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['median']['abs_rel']

    swin2_kitti_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['median']['delta1']
    swin2_nyu_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['median']['delta1']
    swin2_scannet_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['median']['delta1']
    swin2_diode_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['median']['delta1']
    swin2_eth3d_median_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['median']['delta1']

    swin2_kitti_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['global']['abs_rel']
    swin2_nyu_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['global']['abs_rel']
    swin2_scannet_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['global']['abs_rel']
    swin2_diode_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['global']['abs_rel']
    swin2_eth3d_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['global']['abs_rel']

    swin2_kitti_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 0])['global']['delta1']
    swin2_nyu_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 1])['global']['delta1']
    swin2_scannet_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 2])['global']['delta1']
    swin2_diode_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 3])['global']['delta1']
    swin2_eth3d_global_delta1 = read_evaluation_from_log(log_path_list[row_cnt * 5 + 4])['global']['delta1']
    row_cnt +=1


    # print results
    print('**********Final Evaluation Results of AbsRel and delta1 are shown as follows**********\nWe show the affine-invariant depth evaluation results, which uses the least squere repression to align with ground-truth depth and recover the scale and shift values of predicted depth.')

    table_data = [
        ["Method", "KITTI", "KITTI", "NYU", "NYU", "ScanNet", "ScanNet", "DIODE", "DIODE", "ETH3D", "ETH3D"],
        ["", "AbsRel", "delta1", "AbsRel", "delta1", "AbsRel", "delta1", "AbsRel", "delta1", "AbsRel", "delta1"],
        ["Ours_BEiT(median)", beit_kitti_median_absrel, beit_kitti_median_delta1, beit_nyu_median_absrel, beit_nyu_median_delta1, beit_scannet_median_absrel, beit_scannet_median_delta1, beit_diode_median_absrel, beit_diode_median_delta1, beit_eth3d_median_absrel, beit_eth3d_median_delta1],
        ["Ours_ConvNext(median)", convnext_kitti_median_absrel, convnext_kitti_median_delta1, convnext_nyu_median_absrel, convnext_nyu_median_delta1, convnext_scannet_median_absrel, convnext_scannet_median_delta1, convnext_diode_median_absrel, convnext_diode_median_delta1, convnext_eth3d_median_absrel, convnext_eth3d_median_delta1],
        ["Ours_Swin2(median)", swin2_kitti_median_absrel, swin2_kitti_median_delta1, swin2_nyu_median_absrel, swin2_nyu_median_delta1, swin2_scannet_median_absrel, swin2_scannet_median_delta1, swin2_diode_median_absrel, swin2_diode_median_delta1, swin2_eth3d_median_absrel, swin2_eth3d_median_delta1],
        ["Ours_BEiT(global)", beit_kitti_global_absrel, beit_kitti_global_delta1, beit_nyu_global_absrel, beit_nyu_global_delta1, beit_scannet_global_absrel, beit_scannet_global_delta1, beit_diode_global_absrel, beit_diode_global_delta1, beit_eth3d_global_absrel, beit_eth3d_global_delta1],
        ["Ours_ConvNext(global)", convnext_kitti_global_absrel, convnext_kitti_global_delta1, convnext_nyu_global_absrel, convnext_nyu_global_delta1, convnext_scannet_global_absrel, convnext_scannet_global_delta1, convnext_diode_global_absrel, convnext_diode_global_delta1, convnext_eth3d_global_absrel, convnext_eth3d_global_delta1],
        ["Ours_Swin2(global)", swin2_kitti_global_absrel, swin2_kitti_global_delta1, swin2_nyu_global_absrel, swin2_nyu_global_delta1, swin2_scannet_global_absrel, swin2_scannet_global_delta1, swin2_diode_global_absrel, swin2_diode_global_delta1, swin2_eth3d_global_absrel, swin2_eth3d_global_delta1],
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

