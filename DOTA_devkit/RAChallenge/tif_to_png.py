import os
from PIL import Image

original_path = '/data-input/RotationDet/data/RAChallenge/stage1/all_data/images'
saved_path = '/data-input/RotationDet/data/RAChallenge/stage1/all_data/images_png'

counts = 0
files = os.listdir(original_path)
for file in files:
    if file.endswith('tif'):
        tif_file = os.path.join(original_path, file)

        file = file[:-3] + 'png'
        png_file = os.path.join(saved_path, file)
        im = Image.open(tif_file)
        im.save(png_file)
        print(png_file)
        counts += 1

print('%d done' %counts)