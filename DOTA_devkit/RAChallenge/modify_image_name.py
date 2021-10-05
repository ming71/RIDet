import os
import re
import sys

image_path = '/data-input/RotationDet/data/RAChallenge/stage1/all_data_merge/images'
# label_path = '/data-input/RotationDet/data/RAChallenge/stage1/labels'
fileList = os.listdir(image_path)
# 输出此文件夹中包含的文件名称
print("修改前：" + str(fileList)[1])
# 得到进程当前工作目录
currentpath = os.getcwd()
# 将当前工作目录修改为待修改文件夹的位置
os.chdir(image_path)
i = 2008
# 遍历文件夹中所有文件
for fileName in fileList:
    i += 1
    # 匹配文件名正则表达式
    pat = ".+\.(txt)"
    # 进行匹配
    pattern = re.findall(pat, fileName)
    # 文件重新命名
    # print(fileName)
    os.rename(fileName, str(i) + '.tif')
print("***************************************")
# 改回程序运行前的工作目录
os.chdir(currentpath)
# 刷新
sys.stdin.flush()
