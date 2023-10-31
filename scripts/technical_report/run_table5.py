

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
        'bash scripts/ablation/test/test_ablation_data_6_datasets_sevenscenes.sh', # 0, data_6_datasets, sevenscenes
        'bash scripts/ablation/test/test_ablation_data_6_datasets_eth3d.sh', # 1, data_6_datasets, eth3d
        'bash scripts/ablation/test/test_ablation_data_6_datasets_ibims.sh', # 2, data_6_datasets, ibims
        'bash scripts/ablation/test/test_ablation_data_6_datasets_nuscenes.sh', # 3, data_6_datasets, nuscenes

        'bash scripts/ablation/test/test_ablation_data_13_datasets_sevenscenes.sh', # 4, data_13_datasets, sevenscenes
        'bash scripts/ablation/test/test_ablation_data_13_datasets_eth3d.sh', # 5, data_13_datasets, eth3d
        'bash scripts/ablation/test/test_ablation_data_13_datasets_ibims.sh', # 6, data_13_datasets, ibims
        'bash scripts/ablation/test/test_ablation_data_13_datasets_nuscenes.sh', # 7, data_13_datasets, nuscenes
        
        'bash scripts/ablation/test/test_ablation_full_sevenscenes.sh', # 8, full, sevenscenes
        'bash scripts/ablation/test/test_ablation_full_eth3d.sh', # 9, full, eth3d
        'bash scripts/ablation/test/test_ablation_full_ibims.sh', # 10, full, ibims
        'bash scripts/ablation/test/test_ablation_full_nuscenes.sh', # 11, full, nuscenes
    ]

    log_path_list = [
        osp.join(log_root, 'ablation', 'log_ablation_data_6_datasets_sevenscenes.txt'),# 0, data_6_datasets, sevenscenes
        osp.join(log_root, 'ablation', 'log_ablation_data_6_datasets_eth3d.txt'), # 1, data_6_datasets, eth3d
        osp.join(log_root, 'ablation', 'log_ablation_data_6_datasets_ibims.txt'), # 2, data_6_datasets, ibims
        osp.join(log_root, 'ablation', 'log_ablation_data_6_datasets_nuscenes.txt'), # 3, data_6_datasets, nuscenes

        osp.join(log_root, 'ablation', 'log_ablation_data_13_datasets_sevenscenes.txt'),# 4, data_13_datasets, sevenscenes
        osp.join(log_root, 'ablation', 'log_ablation_data_13_datasets_eth3d.txt'), # 5, data_13_datasets, eth3d
        osp.join(log_root, 'ablation', 'log_ablation_data_13_datasets_ibims.txt'), # 6, data_13_datasets, ibims
        osp.join(log_root, 'ablation', 'log_ablation_data_13_datasets_nuscenes.txt'), # 7, data_13_datasets, nuscenes

        osp.join(log_root, 'ablation', 'log_ablation_full_sevenscenes.txt'), # 8, full, sevenscenes
        osp.join(log_root, 'ablation', 'log_ablation_full_eth3d.txt'), # 9, full, eth3d
        osp.join(log_root, 'ablation', 'log_ablation_full_ibims.txt'), # 10, full, ibims
        osp.join(log_root, 'ablation', 'log_ablation_full_nuscenes.txt'), # 11, full,nuscenes
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


    # data_6_datasets
    data_6_datasets_sevenscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 0])['metric']['abs_rel']
    data_6_datasets_eth3d_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 1])['metric']['abs_rel']
    data_6_datasets_ibims_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 2])['metric']['abs_rel']
    data_6_datasets_nuscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 3])['metric']['abs_rel']

    data_6_datasets_sevenscenes_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 0])['global']['abs_rel']
    data_6_datasets_eth3d_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 1])['global']['abs_rel']
    data_6_datasets_ibims_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 2])['global']['abs_rel']
    data_6_datasets_nuscenes_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 3])['global']['abs_rel']
    row_cnt +=1


    # data_13_datasets
    data_13_datasets_sevenscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 0])['metric']['abs_rel']
    data_13_datasets_eth3d_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 1])['metric']['abs_rel']
    data_13_datasets_ibims_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 2])['metric']['abs_rel']
    data_13_datasets_nuscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 3])['metric']['abs_rel']

    data_13_datasets_sevenscenes_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 0])['global']['abs_rel']
    data_13_datasets_eth3d_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 1])['global']['abs_rel']
    data_13_datasets_ibims_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 2])['global']['abs_rel']
    data_13_datasets_nuscenes_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 3])['global']['abs_rel']
    row_cnt +=1


    # full
    full_sevenscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 0])['metric']['abs_rel']
    full_eth3d_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 1])['metric']['abs_rel']
    full_ibims_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 2])['metric']['abs_rel']
    full_nuscenes_metric_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 3])['metric']['abs_rel']

    full_sevenscenes_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 0])['global']['abs_rel']
    full_eth3d_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 1])['global']['abs_rel']
    full_ibims_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 2])['global']['abs_rel']
    full_nuscenes_global_absrel = read_evaluation_from_log(log_path_list[row_cnt * 4 + 3])['global']['abs_rel']
    row_cnt +=1


    # print results
    print('**********Final Evaluation Results of AbsRel are shown as follows**********\nThe "Metric" means we do not use any alignment with ground-truth depth. \nThe "Affine" means we use the least squere repression to align with ground-truth depth and recover the scale and shift values of predicted depth.')

    table_data = [
        ["Method", "Metric", "Metric", "Metric", "Metric", "Affine", "Affine", "Affine", "Affine"],
        ["", "7Scenes", "ETH3D", "iBIMS", "NuScenes", "7Scenes", "ETH3D", "iBIMS", "NuScenes"],
        ["6 datasets", data_6_datasets_sevenscenes_metric_absrel, data_6_datasets_eth3d_metric_absrel, data_6_datasets_ibims_metric_absrel, data_6_datasets_nuscenes_metric_absrel, data_6_datasets_sevenscenes_global_absrel, data_6_datasets_eth3d_global_absrel, data_6_datasets_ibims_global_absrel, data_6_datasets_nuscenes_global_absrel],
        ["13 datasets", data_13_datasets_sevenscenes_metric_absrel, data_13_datasets_eth3d_metric_absrel, data_13_datasets_ibims_metric_absrel, data_13_datasets_nuscenes_metric_absrel, data_13_datasets_sevenscenes_global_absrel, data_13_datasets_eth3d_global_absrel, data_13_datasets_ibims_global_absrel, data_13_datasets_nuscenes_global_absrel],
        ["Full",  full_sevenscenes_metric_absrel, full_eth3d_metric_absrel, full_ibims_metric_absrel, full_nuscenes_metric_absrel, full_sevenscenes_global_absrel, full_eth3d_global_absrel, full_ibims_global_absrel, full_nuscenes_global_absrel],
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

