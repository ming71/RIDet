#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS="4"

ROOT="DOTA"
# ROOT="UCAS_AOD"
# ROOT="HRSC2016"
# ROOT="ICDAR2015"
# ROOT="MSRA_TD500"



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



## single GPU
# python tools/train.py configs/$ROOT/$CONFIG.py

# Multiple GPU
./tools/dist_train.sh configs/$ROOT/$CONFIG.py $GPUS


