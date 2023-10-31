

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
        'bash scripts/test/beit/test_beit_sevenscenes.sh', # 3, beit, sevenscenes
        'bash scripts/test/beit/test_beit_diode_indoor.sh', # 4, beit, diode_indoor
        'bash scripts/test/beit/test_beit_diode_outdoor.sh', # 5, beit, diode_outdoor
        'bash scripts/test/beit/test_beit_diode.sh', # 6, beit, diode
        'bash scripts/test/beit/test_beit_eth3d.sh', # 7, beit, eth3d
        'bash scripts/test/beit/test_beit_ibims.sh', # 8, beit, ibims
        'bash scripts/test/beit/test_beit_nuscenes.sh', # 9, beit, nuscenes

        'bash scripts/test/convnext/test_convnext_kitti.sh', # 10, convnext, kitti
        'bash scripts/test/convnext/test_convnext_nyu.sh', # 11, convnext, nyu
        'bash scripts/test/convnext/test_convnext_scannet.sh', # 12, convnext, scannet
        'bash scripts/test/convnext/test_convnext_sevenscenes.sh', # 13, convnext, sevenscenes
        'bash scripts/test/convnext/test_convnext_diode_indoor.sh', # 14, convnext, diode_indoor
        'bash scripts/test/convnext/test_convnext_diode_outdoor.sh', # 15, convnext, diode_outdoor
        'bash scripts/test/convnext/test_convnext_diode.sh', # 16, convnext, diode
        'bash scripts/test/convnext/test_convnext_eth3d.sh', # 17, convnext, eth3d
        'bash scripts/test/convnext/test_convnext_ibims.sh', # 18, convnext, ibims
        'bash scripts/test/convnext/test_convnext_nuscenes.sh', # 19, convnext, nuscenes

        'bash scripts/test/swin2/test_swin2_kitti.sh', # 20, swin2, kitti
        'bash scripts/test/swin2/test_swin2_nyu.sh', # 21, swin2, nyu
        'bash scripts/test/swin2/test_swin2_scannet.sh', # 22, swin2, scannet
        'bash scripts/test/swin2/test_swin2_sevenscenes.sh', # 23, swin2, sevenscenes
        'bash scripts/test/swin2/test_swin2_diode_indoor.sh', # 24, swin2, diode_indoor
        'bash scripts/test/swin2/test_swin2_diode_outdoor.sh', # 25, swin2, diode_outdoor
        'bash scripts/test/swin2/test_swin2_diode.sh', # 26, swin2, diode
        'bash scripts/test/swin2/test_swin2_eth3d.sh', # 27, swin2, eth3d
        'bash scripts/test/swin2/test_swin2_ibims.sh', # 28, swin2, ibims
        'bash scripts/test/swin2/test_swin2_nuscenes.sh', # 29, swin2, nuscenes
    ]

    log_path_list = [
        osp.join(log_root, 'test', 'log_test_beit_kitti.txt'),# 0, beit, kitti
        osp.join(log_root, 'test', 'log_test_beit_nyu.txt'), # 1, beit, nyu
        osp.join(log_root, 'test', 'log_test_beit_scannet.txt'), # 2, beit, scannet
        osp.join(log_root, 'test', 'log_test_beit_sevenscenes.txt'), # 3, beit, sevenscenes
        osp.join(log_root, 'test', 'log_test_beit_diode_indoor.txt'), # 4, beit, diode_indoor
        osp.join(log_root, 'test', 'log_test_beit_diode_outdoor.txt'), # 5, beit, diode_outdoor
        osp.join(log_root, 'test', 'log_test_beit_diode.txt'), # 6, beit, diode
        osp.join(log_root, 'test', 'log_test_beit_eth3d.txt'), # 7, beit, eth3d
        osp.join(log_root, 'test', 'log_test_beit_ibims.txt'), # 8, beit, ibims
        osp.join(log_root, 'test', 'log_test_beit_nuscenes.txt'), # 9, beit, nuscenes

        osp.join(log_root, 'test', 'log_test_convnext_kitti.txt'),# 10, convnext, kitti
        osp.join(log_root, 'test', 'log_test_convnext_nyu.txt'), # 11, convnext, nyu
        osp.join(log_root, 'test', 'log_test_convnext_scannet.txt'), # 12, convnext, scannet
        osp.join(log_root, 'test', 'log_test_convnext_sevenscenes.txt'), # 13, convnext, sevenscenes
        osp.join(log_root, 'test', 'log_test_convnext_diode_indoor.txt'), # 14, convnext, diode_indoor
        osp.join(log_root, 'test', 'log_test_convnext_diode_outdoor.txt'), # 15, convnext, diode_outdoor
        osp.join(log_root, 'test', 'log_test_convnext_diode.txt'), # 16, convnext, diode
        osp.join(log_root, 'test', 'log_test_convnext_eth3d.txt'), # 17, convnext, eth3d
        osp.join(log_root, 'test', 'log_test_convnext_ibims.txt'), # 18, convnext, ibims
        osp.join(log_root, 'test', 'log_test_convnext_nuscenes.txt'), # 19, convnext, nuscenes

        osp.join(log_root, 'test', 'log_test_swin2_kitti.txt'),# 20, swin2, kitti
        osp.join(log_root, 'test', 'log_test_swin2_nyu.txt'), # 21, swin2, nyu
        osp.join(log_root, 'test', 'log_test_swin2_scannet.txt'), # 22, swin2, scannet
        osp.join(log_root, 'test', 'log_test_swin2_sevenscenes.txt'), # 23, swin2, sevenscenes
        osp.join(log_root, 'test', 'log_test_swin2_diode_indoor.txt'), # 24, swin2, diode_indoor
        osp.join(log_root, 'test', 'log_test_swin2_diode_outdoor.txt'), # 25, swin2, diode_outdoor
        osp.join(log_root, 'test', 'log_test_swin2_diode.txt'), # 26, swin2, diode
        osp.join(log_root, 'test', 'log_test_swin2_eth3d.txt'), # 27, swin2, eth3d
        osp.join(log_root, 'test', 'log_test_swin2_ibims.txt'), # 28, swin2, ibims
        osp.join(log_root, 'test', 'log_test_swin2_nuscenes.txt'), # 29, swin2, nuscenes
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
    beit_kitti_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 0])['median']['abs_rel']
    beit_nyu_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 1])['median']['abs_rel']
    beit_scannet_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 2])['median']['abs_rel']
    beit_sevenscenes_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 3])['median']['abs_rel']
    beit_diode_indoor_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 4])['median']['abs_rel']
    beit_diode_outdoor_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 5])['median']['abs_rel']
    beit_diode_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 6])['median']['abs_rel']
    beit_eth3d_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 7])['median']['abs_rel']
    beit_ibims_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 8])['median']['abs_rel']
    beit_nuscenes_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 9])['median']['abs_rel']

    row_cnt +=1


    # convnext
    convnext_kitti_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 0])['median']['abs_rel']
    convnext_nyu_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 1])['median']['abs_rel']
    convnext_scannet_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 2])['median']['abs_rel']
    convnext_sevenscenes_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 3])['median']['abs_rel']
    convnext_diode_indoor_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 4])['median']['abs_rel']
    convnext_diode_outdoor_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 5])['median']['abs_rel']
    convnext_diode_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 6])['median']['abs_rel']
    convnext_eth3d_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 7])['median']['abs_rel']
    convnext_ibims_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 8])['median']['abs_rel']
    convnext_nuscenes_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 9])['median']['abs_rel']
    row_cnt +=1


    # swin2
    swin2_kitti_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 0])['median']['abs_rel']
    swin2_nyu_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 1])['median']['abs_rel']
    swin2_scannet_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 2])['median']['abs_rel']
    swin2_sevenscenes_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 3])['median']['abs_rel']
    swin2_diode_indoor_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 4])['median']['abs_rel']
    swin2_diode_outdoor_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 5])['median']['abs_rel']
    swin2_diode_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 6])['median']['abs_rel']
    swin2_eth3d_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 7])['median']['abs_rel']
    swin2_ibims_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 8])['median']['abs_rel']
    swin2_nuscenes_median_abs_rel = read_evaluation_from_log(log_path_list[row_cnt * 10 + 9])['median']['abs_rel']
    row_cnt +=1


    # print results
    print('**********Final Evaluation Results of AbsRel are shown as follows**********\nWe show the median depth evaluation results and align the predictied to the ground-truth depth by multiplying a median value ratio.')

    table_data = [
        ["Method", "KITTI", "NYU", "ScanNet", "7-Scenes", "DIODE(indoor)", "DIODE(outdoor)", "DIODE", "ETH3D", "iBIMS-1", "NuScenes"],
        ["Ours_BEiT", beit_kitti_median_abs_rel, beit_nyu_median_abs_rel, beit_scannet_median_abs_rel, beit_sevenscenes_median_abs_rel, beit_diode_indoor_median_abs_rel, beit_diode_outdoor_median_abs_rel, beit_diode_median_abs_rel, beit_eth3d_median_abs_rel, beit_ibims_median_abs_rel, beit_nuscenes_median_abs_rel],
        ["Ours_ConvNext", convnext_kitti_median_abs_rel, convnext_nyu_median_abs_rel, convnext_scannet_median_abs_rel, convnext_sevenscenes_median_abs_rel, convnext_diode_indoor_median_abs_rel, convnext_diode_outdoor_median_abs_rel, convnext_diode_median_abs_rel, convnext_eth3d_median_abs_rel, convnext_ibims_median_abs_rel, convnext_nuscenes_median_abs_rel],
        ["Ours_Swin2", swin2_kitti_median_abs_rel, swin2_nyu_median_abs_rel, swin2_scannet_median_abs_rel, swin2_sevenscenes_median_abs_rel, swin2_diode_indoor_median_abs_rel, swin2_diode_outdoor_median_abs_rel, swin2_diode_median_abs_rel, swin2_eth3d_median_abs_rel, swin2_ibims_median_abs_rel, swin2_nuscenes_median_abs_rel],
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
