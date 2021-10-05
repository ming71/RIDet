import cv2, os, argparse
import numpy as np
from tqdm import tqdm
import numpy as np
# from scipy import misc
from PIL import Image
# from libtiff import TIFF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    img_filenames = os.listdir(opt.dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        # tiff
        # tif = TIFF.open(opt.dir + '/' + img_filename, mode='r')
        # img = tif.read_image()  # 此时img为一个numpy.array

        # tiff
        tif = Image.open(opt.dir + '/' + img_filename, mode='r')
        img = np.array(tif)

        # img = cv2.imread(opt.dir + '/' + img_filename)
        # print(img)

        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])

if __name__ == '__main__':
    main()

    # --dir /home/alex/object_detection_code/RotationDet/data/HRSC/Train/images
