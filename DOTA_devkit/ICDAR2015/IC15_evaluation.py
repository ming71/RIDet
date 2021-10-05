import os
import cv2
import shutil
import zipfile
import argparse
import os.path as osp

def make_zip(source_dir, output_file):
    zipf = zipfile.ZipFile(output_file, 'w')
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            zipf.write(pathfile, filename)
    zipf.close()


def eval_IC15(dets_dir, eval_dir):
    zip_name = 'submit.zip'
    zip_file = osp.join(eval_dir, zip_name)
    
    make_zip(dets_dir, zip_file)
    # shutil.move(os.path.join('./', zip_name), zip_file)
    # if os.path.exists(dets_dir):
    #     shutil.rmtree(dets_dir)
    result = os.popen('cd {0} && python script.py -g=gt.zip -s=submit.zip '.format(eval_dir)).read()
    sep = result.split(':')
    precision = sep[1][:sep[1].find(',')].strip()
    recall = sep[2][:sep[2].find(',')].strip()
    f1 = sep[3][:sep[3].find(',')].strip()
    p = eval(precision)
    r = eval(recall)
    hmean = eval(f1)
    pf = '%15s' + '%10.3g' + '%15s' + '%10.3g' +'%15s' + '%10.3g'   # print format
    print(pf % ('P:', p, 'R:', r, 'F1:', hmean))

def parse_args():
    parser = argparse.ArgumentParser(description='IC15 evaluation')
    parser.add_argument('config', default='configs/DOTA/faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5.py')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    config_name = osp.splitext(osp.basename(config_file))[0]
    work_dir = osp.join('work_dirs', config_name)
    dets_dir = osp.join(work_dir, 'det_results')
    eval_IC15(dets_dir, 'DOTA_devkit/ICDAR2015/eval')
