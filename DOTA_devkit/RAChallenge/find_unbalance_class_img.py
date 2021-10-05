import xml.etree.ElementTree as et
import os
from tqdm import tqdm
import shutil
import cv2
import numpy as np


def find_unbalce_class_img():
    data_dir = '/data-input/RotationDet/data/RAChallenge/warmup/'
    target_dir = '/data-input/RotationDet/data/RAChallenge/air_carrier/'
    data_label_dir = data_dir + 'labelTxt/'
    data_image_dir = data_dir + 'images/'
    target_label_dir = target_dir + 'labelTxt/'
    target_image_dir = target_dir + 'images/'
    files = os.listdir(data_label_dir)
    i = 25
    for filename in tqdm(files):
        with open(os.path.join(data_label_dir, filename), 'r') as f:
            now_img_file = os.path.join(data_image_dir, filename[:-4] + '.tif')
            now_txt_file = os.path.join(data_label_dir, filename[:-4] + '.txt')
            for line in f:
                if line.split(" ")[-2] == '1':
                    for x in range(0, 1):
                        i += 1
                        targer_img_file = os.path.join(target_image_dir, str(i) + '.tif')
                        targer_txt_file = os.path.join(target_label_dir, str(i) + '.txt')
                        shutil.copy(now_txt_file, targer_txt_file)
                        shutil.copy(now_img_file, targer_img_file)


            # objs = contents.split('<object>')[1:]
            # for obj in objs:
            #     classname = obj[obj.find('<name>') + 6: obj.find('</name>')]
            #     # if classname == 'Boeing777':
            #     if classname == 'Boeing777' or classname == 'ARJ21':
            #         i += 1
            #         targer_img_file = os.path.join(target_img_dir_path, str(i) + '.png')
            #         targer_txt_file = os.path.join(target_xml_dir_path, str(i) + '.txt')
            #
            #         # Guassion process
            #         # img = cv2.imread(now_img_file)  # BGR
            #         # noise = np.random.normal(0, 5, img.shape)
            #         # img = img + noise.astype(int)  # 高斯噪声转为int型便于相加
            #         # np.maximum(img, 0)  # 规范到0-255内
            #         # np.minimum(img, 255)
            #         # cv2.imwrite(targer_img_file, img)
            #
            #         shutil.copy(now_txt_file, targer_txt_file)
            #         shutil.copy(now_img_file, targer_img_file)
            
            
def select_air_carrier_label():
    target_dir = '/data-input/RotationDet/data/RAChallenge/air_carrier/'
    target_label_dir = target_dir + 'labelTxt/'
    files = os.listdir(target_label_dir)
    for filename in tqdm(files):
        with open(os.path.join(target_label_dir, filename), 'r') as f:
            lines = f.readlines()
            # for line in lines:
            #     print(line.split(" ")[0] == '1')
        with open(os.path.join(target_label_dir, filename), 'w') as f_w:
            for line in lines:
                if line.split(" ")[-2] == '1':
                    f_w.write(line)

                    
def delete_air_carrier_submission():
    target_dir = '/home/alex/下载/科目四_BITRS.txt'
    with open(target_dir, 'r') as f:
        lines = f.readlines()
        # for line in lines:
        #     print(line.split(" ")[0] == '1')
    with open(target_dir, 'w') as f_w:
            for line in lines:
                print()
                if line.split(" ")[1] != '1':
                    f_w.write(line)

def delete_air_carrier_label():
    data_dir = '/data-input/RotationDet/data/RAChallenge/stage1/all_data_merge/'
    data_img_dir = data_dir + 'images/'
    data_label_dir = data_dir + 'labels/'
    files = os.listdir(data_label_dir)
    for filename in tqdm(files):
        with open(os.path.join(data_label_dir, filename), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(data_label_dir, filename), 'w') as f_w:
                for line in lines:
                    if line.split(" ")[0] != '1':
                        f_w.write(line)
        with open(os.path.join(data_label_dir, filename), 'r') as f:
            lines = f.readlines()
            if not lines:
                print(os.path.join(data_label_dir, filename))
                print(os.path.join(data_img_dir, filename[:-4] + ".tif"))
                os.remove(os.path.join(data_label_dir, filename))
                os.remove(os.path.join(data_img_dir, filename[:-4] + ".tif"))




if __name__ == '__main__':
#     find_unbalce_class_img()
#     select_air_carrier_label()
    delete_air_carrier_label()