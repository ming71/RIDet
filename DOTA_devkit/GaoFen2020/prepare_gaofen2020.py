import os
import sys
import glob
import shutil
import os.path as osp
sys.path.append("..")
from GAOFEN2DOTA import generate_txt_labels
from GAOFEN2JSON import generate_json_labels

def preprare_gaofen2020_airplane(data_dir):
    print('Prepare GaoFen2020 Airplane')
    train_dir = osp.join(data_dir,'train')
    test_dir = osp.join(data_dir, 'val')
    # convert xml to dota  format
    generate_txt_labels(train_dir)
    # convert it to json format
    generate_json_labels(train_dir,osp.join(train_dir,'train.json'), type='plane')
    generate_json_labels(test_dir,osp.join(test_dir,'test.json'), trainval=False, type='plane')

def make_clean_dir(folder):
    if osp.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def copyfiles(files, dst_folder):
    for src in files:
        dst = osp.join(dst_folder, osp.split(src)[1])
        shutil.move(src, dst)

def preprare_gaofen2020_ship(data_dir):
    print('Prepare GaoFen2020 Ship')
    train_dir = osp.join(data_dir,'train')
    test_dir = osp.join(data_dir, 'val')
    train_im_dir = osp.join(train_dir,'images')
    train_an_dir = osp.join(train_dir,'label_xml')
    test_im_dir = osp.join(test_dir,'images')
    dirs = [train_im_dir, train_an_dir, test_im_dir]
    # build tree
    for folder in dirs:
        make_clean_dir(folder)
    train_ims = glob.glob(train_dir+'/*.tiff')
    train_ans = glob.glob(train_dir+'/*.xml')
    test_ims  = glob.glob(test_dir+'/*.tiff')
    copyfiles(train_ims, train_im_dir)
    copyfiles(train_ans, train_an_dir)
    copyfiles(test_ims,  test_im_dir)
    # convert format
    generate_txt_labels(train_dir)
    generate_json_labels(train_dir,osp.join(train_dir,'train.json'), type='ship')
    generate_json_labels(test_dir,osp.join(test_dir,'test.json'), trainval=False, type='ship')

if __name__ == '__main__':
    gaofen2020_dir = '/data-input/RotationDet/data/GaoFen2020'
    gaofen2020_airplane = osp.join(gaofen2020_dir, 'airplane')
    gaofen2020_ship = osp.join(gaofen2020_dir, 'ship')

    # preprare_gaofen2020_airplane(gaofen2020_airplane)
    preprare_gaofen2020_ship(gaofen2020_ship)

    print('done')