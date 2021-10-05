import os
import sys
import json
import shutil
import xmltodict
import numpy as np
import os.path as osp
from tqdm import tqdm

sys.path.append("..")
from dota_poly2rbox import rbox2poly_single


def parse_ann_info(objects):
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        bbox = [eval(x) for x in ','.join(obj['points']['point']).split(',')]
        label = obj['possibleresult']['name']
        bboxes.append(bbox)
        labels.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore


def ann_to_txt(ann):
    out_str = ''
    for bbox, label in zip(ann['bboxes'], ann['labels']):
        poly = np.array(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '0')
        out_str += str_line
    for bbox, label in zip(ann['bboxes_ignore'], ann['labels_ignore']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '1')
        out_str += str_line
    return out_str
    


def generate_txt_labels(root_path):
    img_path = osp.join(root_path, 'images')
    label_path = osp.join(root_path, 'label_xml')
    label_txt_path = osp.join(root_path, 'labelTxt')
    if  osp.exists(label_txt_path):
        shutil.rmtree(label_txt_path)
    os.mkdir(label_txt_path)

    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    pbar = tqdm(img_names)
    for img_name in pbar:
        pbar.set_description("GaoFen2020 generate_txt_labels")
        label = osp.join(label_path, img_name+'.xml')
        label_txt = osp.join(label_txt_path, img_name+'.txt')
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['annotation']
        f_label.close()
        label_txt_str = ''
        # with annotations
        if data_dict['objects']:
            objects = data_dict['objects']['object']
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(
                objects)
            ann = dict(
                bboxes=bboxes,
                labels=labels,
                bboxes_ignore=bboxes_ignore,
                labels_ignore=labels_ignore)
            label_txt_str = ann_to_txt(ann)
        with open(label_txt,'w') as f_txt:
            f_txt.write(label_txt_str)

if __name__ == '__main__':
    generate_txt_labels('/data-input/RotationDet/data/GaoFen2020/airplane/train')
    print('done!')
