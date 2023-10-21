from html4vision import Col, imagetable
import os
import os.path as osp

def create_html(name2list, save_path='index.html'):
    # table description
    cols = []
    for k, v in name2list.items():
        col_i = Col('img', k, v)
        cols.append(col_i)
    
    imagetable(cols, out_file=save_path, imsize=(256, 384))

if __name__ == '__main__':
    data_dir = './temp'
    rgbs = osp.join(data_dir, '*_rgb.jpg')
    pred = osp.join(data_dir, '*_pred.jpg')
    gt = osp.join(data_dir, '*_gt.jpg')

    name_path = dict(rgbs=rgbs, pred=pred, gt=gt)
    html = create_html(name_path, save_path=data_dir + '_view.html')
    