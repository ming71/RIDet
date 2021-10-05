#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

./DOTA_devkit/compute_img_mean_std.py --dir data-input/RotationDet/data/dota_1024/trainval_split/images
