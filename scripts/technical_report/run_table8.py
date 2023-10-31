

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
        'bash scripts/test/beit/test_beit_diode_indoor.sh', # 0, beit, diode_indoor
        'bash scripts/test/beit/test_beit_ibims.sh', # 1, beit, ibims
        'bash scripts/test/beit/test_beit_sevenscenes.sh', # 2, beit, sevenscenes
        'bash scripts/test/beit/test_beit_diode_outdoor.sh', # 3, beit, diode_outdoor
        'bash scripts/test/beit/test_beit_eth3d.sh', # 4, beit, eth3d
        'bash scripts/test/beit/test_beit_nuscenes.sh', # 5, beit, nuscenes

        'bash scripts/test/convnext/test_convnext_diode_indoor.sh', # 6, convnext, diode_indoor
        'bash scripts/test/convnext/test_convnext_ibims.sh', # 7, convnext, ibims
        'bash scripts/test/convnext/test_convnext_sevenscenes.sh', # 8, convnext, sevenscenes
        'bash scripts/test/convnext/test_convnext_diode_outdoor.sh', # 9, convnext, diode_outdoor
        'bash scripts/test/convnext/test_convnext_eth3d.sh', # 10, convnext, eth3d
        'bash scripts/test/convnext/test_convnext_nuscenes.sh', # 11, convnext, nuscenes

        'bash scripts/test/swin2/test_swin2_diode_indoor.sh', # 12, swin2, diode_indoor
        'bash scripts/test/swin2/test_swin2_ibims.sh', # 13, swin2, ibims
        'bash scripts/test/swin2/test_swin2_sevenscenes.sh', # 14, swin2, sevenscenes
        'bash scripts/test/swin2/test_swin2_diode_outdoor.sh', # 15, swin2, diode_outdoor
        'bash scripts/test/swin2/test_swin2_eth3d.sh', # 16, swin2, eth3d
        'bash scripts/test/swin2/test_swin2_nuscenes.sh', # 17, swin2, nuscenes
    ]

    log_path_list = [
        osp.join(log_root, 'test', 'log_test_beit_diode_indoor.txt'),# 0, beit, diode_indoor
        osp.join(log_root, 'test', 'log_test_beit_ibims.txt'), # 1, beit, ibims
        osp.join(log_root, 'test', 'log_test_beit_sevenscenes.txt'), # 2, beit, sevenscenes
        osp.join(log_root, 'test', 'log_test_beit_diode_outdoor.txt'), # 3, beit, diode_outdoor
        osp.join(log_root, 'test', 'log_test_beit_eth3d.txt'), # 4, beit, eth3d
        osp.join(log_root, 'test', 'log_test_beit_nuscenes.txt'), # 5, beit, nuscenes

        osp.join(log_root, 'test', 'log_test_convnext_diode_indoor.txt'),# 6, convnext, diode_indoor
        osp.join(log_root, 'test', 'log_test_convnext_ibims.txt'), # 7, convnext, ibims
        osp.join(log_root, 'test', 'log_test_convnext_sevenscenes.txt'), # 8, convnext, sevenscenes
        osp.join(log_root, 'test', 'log_test_convnext_diode_outdoor.txt'), # 9, convnext, diode_outdoor
        osp.join(log_root, 'test', 'log_test_convnext_eth3d.txt'), # 10, convnext, eth3d
        osp.join(log_root, 'test', 'log_test_convnext_nuscenes.txt'), # 11, convnext, nuscenes

        osp.join(log_root, 'test', 'log_test_swin2_diode_indoor.txt'),# 12, swin2, diode_indoor
        osp.join(log_root, 'test', 'log_test_swin2_ibims.txt'), # 13, swin2, ibims
        osp.join(log_root, 'test', 'log_test_swin2_sevenscenes.txt'), # 14, swin2, sevenscenes
        osp.join(log_root, 'test', 'log_test_swin2_diode_outdoor.txt'), # 15, swin2, diode_outdoor
        osp.join(log_root, 'test', 'log_test_swin2_eth3d.txt'), # 16, swin2, eth3d
        osp.join(log_root, 'test', 'log_test_swin2_nuscenes.txt'), # 17, swin2, nuscenes
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
    beit_diode_indoor_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 0])['metric']['abs_rel']
    beit_ibims_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 1])['metric']['abs_rel']
    beit_sevenscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 2])['metric']['abs_rel']
    beit_diode_outdoor_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 3])['metric']['abs_rel']
    beit_eth3d_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 4])['metric']['abs_rel']
    beit_nuscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 5])['metric']['abs_rel']

    beit_diode_indoor_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 0])['metric']['rmse']
    beit_ibims_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 1])['metric']['rmse']
    beit_sevenscenes_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 2])['metric']['rmse']
    beit_diode_outdoor_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 3])['metric']['rmse']
    beit_eth3d_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 4])['metric']['rmse']
    beit_nuscenes_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 5])['metric']['rmse']
    row_cnt +=1


    # convnext
    convnext_diode_indoor_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 0])['metric']['abs_rel']
    convnext_ibims_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 1])['metric']['abs_rel']
    convnext_sevenscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 2])['metric']['abs_rel']
    convnext_diode_outdoor_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 3])['metric']['abs_rel']
    convnext_eth3d_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 4])['metric']['abs_rel']
    convnext_nuscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 5])['metric']['abs_rel']

    convnext_diode_indoor_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 0])['metric']['rmse']
    convnext_ibims_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 1])['metric']['rmse']
    convnext_sevenscenes_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 2])['metric']['rmse']
    convnext_diode_outdoor_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 3])['metric']['rmse']
    convnext_eth3d_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 4])['metric']['rmse']
    convnext_nuscenes_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 5])['metric']['rmse']
    row_cnt +=1


    # swin2
    swin2_diode_indoor_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 0])['metric']['abs_rel']
    swin2_ibims_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 1])['metric']['abs_rel']
    swin2_sevenscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 2])['metric']['abs_rel']
    swin2_diode_outdoor_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 3])['metric']['abs_rel']
    swin2_eth3d_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 4])['metric']['abs_rel']
    swin2_nuscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 6 + 5])['metric']['abs_rel']

    swin2_diode_indoor_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 0])['metric']['rmse']
    swin2_ibims_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 1])['metric']['rmse']
    swin2_sevenscenes_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 2])['metric']['rmse']
    swin2_diode_outdoor_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 3])['metric']['rmse']
    swin2_eth3d_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 4])['metric']['rmse']
    swin2_nuscenes_metric_rmse = read_evaluation_from_log(log_path_list[row_cnt * 6 + 5])['metric']['rmse']
    row_cnt +=1


    # print results
    print('**********Final Evaluation Results of AbsRel and RMSE are shown as follows**********\nWe show the metric depth evaluation results and do not use any alignment with ground-truth depth.')

    table_data = [
        ["Method", "DIODE(indoor)", "DIODE(indoor)", "iBIMS-1", "iBIMS-1", "7-Scenes", "7-Scenes", "DIODE(outdoor)", "DIODE(outdoor)", "ETH3D", "ETH3D", "NuScenes", "NuScenes"],
        ["", "AbsRel", "RMSE", "AbsRel", "RMSE", "AbsRel", "RMSE", "AbsRel", "RMSE", "AbsRel", "RMSE", "AbsRel", "RMSE"],
        ["Ours_BEiT", beit_diode_indoor_metric_absrel, beit_diode_indoor_metric_rmse, beit_ibims_metric_absrel, beit_ibims_metric_rmse, beit_sevenscenes_metric_absrel, beit_sevenscenes_metric_rmse, beit_diode_outdoor_metric_absrel, beit_diode_outdoor_metric_rmse, beit_eth3d_metric_absrel, beit_eth3d_metric_rmse, beit_nuscenes_metric_absrel, beit_nuscenes_metric_rmse],
        ["Ours_ConvNext", convnext_diode_indoor_metric_absrel, convnext_diode_indoor_metric_rmse, convnext_ibims_metric_absrel, convnext_ibims_metric_rmse, convnext_sevenscenes_metric_absrel, convnext_sevenscenes_metric_rmse, convnext_diode_outdoor_metric_absrel, convnext_diode_outdoor_metric_rmse, convnext_eth3d_metric_absrel, convnext_eth3d_metric_rmse, convnext_nuscenes_metric_absrel, convnext_nuscenes_metric_rmse],
        ["Ours_Swin2", swin2_diode_indoor_metric_absrel, swin2_diode_indoor_metric_rmse, swin2_ibims_metric_absrel, swin2_ibims_metric_rmse, swin2_sevenscenes_metric_absrel, swin2_sevenscenes_metric_rmse, swin2_diode_outdoor_metric_absrel, swin2_diode_outdoor_metric_rmse, swin2_eth3d_metric_absrel, swin2_eth3d_metric_rmse, swin2_nuscenes_metric_absrel, swin2_nuscenes_metric_rmse],
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

