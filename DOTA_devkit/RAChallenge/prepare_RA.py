import os
import sys
import glob
import shutil
import os.path as osp
sys.path.append("..")
from RA2DOTA import generate_txt_labels
from RA2JSON import generate_json_labels

def preprare_RA(data_dir):
#     print('Prepare RA dataset')
#     warmup_dir = osp.join(data_dir,'warmup')
#     warmup_aug_dir = osp.join(data_dir,'warmup_augment')
#     stage1_dir = osp.join(data_dir,'stage1/train')
#     stage1_aug_dir = osp.join(data_dir,'stage1/train_augment')
#     test1_dir = osp.join(data_dir, 'stage1')
#     # convert txt to dota  format
#     generate_txt_labels(warmup_dir)
#     generate_txt_labels(warmup_aug_dir)
#     generate_txt_labels(stage1_dir)
#     generate_txt_labels(stage1_aug_dir)
#     # convert it to json format
#     generate_json_labels(warmup_dir,osp.join(warmup_dir,'train.json'))
#     generate_json_labels(warmup_aug_dir,osp.join(warmup_aug_dir,'train.json'))
#     generate_json_labels(stage1_dir,osp.join(stage1_dir,'train.json'))
#     generate_json_labels(stage1_aug_dir,osp.join(stage1_aug_dir,'train.json'))
#     generate_json_labels(test1_dir,osp.join(test1_dir,'test.json'), trainval=False)

    
      # warmup merge 
#       warmup_merge_dir = osp.join(data_dir,'train_merge')
#       generate_txt_labels(warmup_merge_dir)
#       generate_json_labels(warmup_merge_dir,osp.join(warmup_merge_dir,'train.json'))
      
#       test
#       generate_json_labels(data_dir,osp.join(data_dir,'test.json'), trainval=False)
      # train
        generate_txt_labels(data_dir)
        generate_json_labels(data_dir,osp.join(data_dir,'train.json'))
       
def make_clean_dir(folder):
    if osp.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def copyfiles(files, dst_folder):
    for src in files:
        dst = osp.join(dst_folder, osp.split(src)[1])
        shutil.move(src, dst)

if __name__ == '__main__':
    root_dir = '/data-input/RotationDet/data/RAChallenge/stage1/all_data_augment/'

    preprare_RA(root_dir)

    print('done')