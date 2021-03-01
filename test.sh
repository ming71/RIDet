#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS="4"

# DATASET="HRSC2016"
# DATASET="UCAS_AOD"
# DATASET="ICDAR2015"
DATASET="DOTA"
# DATASET="MSRA_TD500"


CHECKPOINT="epoch_12"


#### HRSC20116
# CONFIG="s2anet_r50_fpn_416_hungarian_hrsc2016"

# CONFIG="cascade_retinanet_obb_r50_fpn_416_hrsc2016"
# CONFIG="cascade_retinanet_obb_r50_fpn_mr_416_hrsc2016"
# CONFIG="cascade_retinanet_obb_r50_fpn_gwd_416_hrsc2016"
# CONFIG="cascade_retinanet_obb_r50_fpn_iou_smooth_416_hrsc2016"
# CONFIG="cascade_retinanet_obb_r50_fpn_hungarian_416_hrsc2016"

# CONFIG="cascade_retinanet_quad_r50_fpn_416_hrsc2016"
# CONFIG="cascade_retinanet_quad_r50_fpn_mr_416_hrsc2016"
# CONFIG="cascade_retinanet_quad_r50_fpn_sbd_416_hrsc2016"
# CONFIG="cascade_retinanet_quad_r50_fpn_gwd_416_hrsc2016"
# CONFIG="cascade_retinanet_quad_r50_fpn_hungarian_416_hrsc2016"
# CONFIG="cascade_retinanet_quad_r50_fpn_glidvertex_416_hrsc2016"

# CONFIG="cascade_retinanet_obb_r101_fpn_hungarian_800_hrsc2016"
# CONFIG="cascade_retinanet_quad_r101_fpn_hungarian_800_hrsc2016"


#### UCAS-AOD
# CONFIG="cascade_retinanet_obb_r50_fpn_hungarian_800_ucas_aod"
# CONFIG="cascade_retinanet_quad_r50_fpn_hungarian_800_ucas_aod"

#### IC15
# CONFIG="cascade_retinanet_quad_r101_fpn_hungarian_800_ic15"

#### MSRA-TD500
# CONFIG="cascade_retinanet_obb_r50_fpn_hungarian_800_msra_td500"

#### DOTA
# CONFIG="s2anet_r50_fpn_hungarian_800_dota"
# CONFIG="cascade_retinanet_obb_r101_fpn_hungarian_800_dota"
CONFIG="cascade_retinanet_quad_r101_fpn_hungarian_800_dota"




if   [ $DATASET = "HRSC2016" ]; then
    DATASET="hrsc2016"
    ROOT="HRSC2016"
    EVAL="hrsc2016_evaluation"

elif [ $DATASET = "UCAS_AOD" ]; then
    DATASET="hrsc2016"
    ROOT="UCAS_AOD"
    EVAL="ucas_evaluation"

elif [ $DATASET = "VOC" ]; then
    DATASET="voc"
    EVAL="tools/voc_eval"

elif [ $DATASET = "ICDAR2015" ]; then
    DATASET="ic15"
    ROOT="ICDAR2015"
    EVAL="IC15_evaluation"

elif [ $DATASET = "DOTA" ]; then
    DATASET="dota"
    ROOT="DOTA"

elif [ $DATASET = "MSRA_TD500" ]; then
    DATASET="msra_td500"
    ROOT="MSRA_TD500"
    EVAL="MSRA_TD500_evaluation"
fi


## single-GPU
# python tools/test.py configs/$ROOT/$CONFIG.py \
#     work_dirs/$CONFIG/$CHECKPOINT.pth \
#     --out work_dirs/$CONFIG/results.pkl \
#     --data $DATASET 

# Mul-GPus
./tools/dist_test.sh configs/$ROOT/$CONFIG.py \
    work_dirs/$CONFIG/$CHECKPOINT.pth  $GPUS \
    --out work_dirs/$CONFIG/results.pkl \
    --data $DATASET 

if   [ $DATASET = "dota" ]; then
    zip -j Task1.zip work_dirs/$CONFIG/result_merge/*
elif [ $DATASET = "voc" ]; then
    python $EVAL.py \
        configs/$ROOT/$CONFIG.py 
else 
    python DOTA_devkit/$ROOT/$EVAL.py \
        configs/$ROOT/$CONFIG.py     
fi
