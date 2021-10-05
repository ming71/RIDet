import os
import sys
import json
import codecs
import shutil
import zipfile
import argparse
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config
from tqdm import tqdm

from mmdet.datasets import build_dataset
from mmdet.core.bbox import rbox2poly_single

from lxml.etree import Element,SubElement,tostring
import xml.etree.ElementTree as ElementTree
from xml.dom.minidom import parseString
import xml.dom.minidom

 
def creat_submission(data_test, dets, dstpath, classnames):
    author = r'科目四_BITRS'
    subname = author + '.txt'

    with open(data_test.ann_file,'r') as ft:
        images = json.load(ft)
        im_names = [osp.splitext(x['filename'])[0] for x in images] 

    submission = '' 
    pbar = tqdm(dets)
    for img_di, det in enumerate(pbar):
        pbar.set_description("creat_submission")
        im_name = im_names[img_di]
        for cls_id, cls_dets in enumerate(det):
            for obj in cls_dets:
                s = ''
                *bbox, conf = obj
                quad_bbox = rbox2poly_single(bbox)
                str_bbox = [str(int(x)) for x in quad_bbox]
                s = im_name + '.tif' + ' ' + classnames[cls_id] + ' ' + str(conf) + ' ' + ' '.join(str_bbox) + '\n'
                submission += s
    fw = codecs.open(osp.join(dstpath, subname), "w", "utf-8" )
    fw.write(submission)
    fw.close()



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', default='configs/DOTA/faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5.py')
    args = parser.parse_args()
    return args

def parse_results(config_file, resultfile, dstpath ):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data['test']
    dataset = build_dataset(data_test)
    outputs = mmcv.load(resultfile)
    dataset_type = cfg.dataset_type
    classnames = dataset.CLASSES
    creat_submission(data_test, outputs, dstpath, classnames)



if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    config_name = osp.splitext(osp.basename(config_file))[0]
    pkl_file = osp.join('work_dirs', config_name, 'results.pkl')
#     pkl_file = 'work_dirs/s2anet_r50_fpn_1x_ms_rotate/results.pkl'
    output_path = osp.join('work_dirs', config_name)
#     output_path = 'work_dirs/s2anet_r50_fpn_1x_ms_rotate/'
    parse_results(config_file, pkl_file, output_path)

