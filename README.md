# Optimization for Arbitrary-Oriented Object Detection via Representation Invariance Loss

<div align=center><img width="500" height="300" src="https://github.com/ming71/RIDet/blob/RIDet-pytorch/pics/RIL.png"/></div>

By [Qi Ming](https://ming71.github.io/), Zhiqiang Zhou, Lingjuan Miao, [Xue Yang](https://yangxue0827.github.io/index.html), and Yunpeng Dong.

The repository hosts the codes for our paper `Optimization for Arbitrary-Oriented Object Detection via Representation Invariance Loss` ([PDF](https://ieeexplore.ieee.org/document/9555916), [arxiv](https://arxiv.org/abs/2103.11636)), based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [s2anet](https://github.com/csuhan/s2anet). 

## Introduction

Arbitrary-oriented objects exist widely in remote sensing images. The mainstream rotation detectors use oriented bounding boxes (OBB) or quadrilateral bounding boxes (QBB) to represent the rotating objects. However, these methods suffer from the representation ambiguity for oriented object definition, which leads to suboptimal regression optimization and the inconsistency between the loss metric and the localization accuracy of the predictions. In this paper, we propose a Representation Invariance Loss (RIL) to optimize the bounding box regression for the rotating objects in the remote sensing images. RIL treats multiple representations of an oriented object as multiple equivalent local minima, and hence transforms bounding box regression into an adaptive matching process with these local minima. Next, the Hungarian matching algorithm is adopted to obtain the optimal regression strategy. Besides, we propose a normalized rotation loss to alleviate the weak correlation between different variables and their unbalanced loss contribution in OBB representation. Extensive experiments on remote sensing datasets show that our method achieves consistent and substantial improvement. 


## Installation
```
conda create -n ridet python=3.7 -y
source activate ridet
conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch

pip install -r requirements.txt
python setup.py develop
cd mmdet/ops/orn
python setup.py build_ext --inplace

apt-get update
apt-get install swig
apt-get install zip

cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
cd ..
```

## Getting Started

### Datasets
* DOTA
* HRSC2016
* ICDAR2015
* UCAS-AOD
* VOC2007
* MSRA-TD500

### Data Preration
```
cd DOTA_devkit/$DATASET
python prepare_$DATASET.py
```

### Training
Set the following configuration according to your own file directory: `$GPUS`, `$ROOT`, `$CONFIG`, and then start training:
```
sh train.sh
```

### Testing
Set the following configuration according to your own file directory: `$GPUS`, `$DATASET`, `$CHECKPOINT`, `$CONFIG`, and then start evaluation:
```
sh test.sh
```


### Demo
To output the visualization of the detections, the following configuration need to be set: `$ROOT`, `$IMAGES`, `$CHECKPOINT`, `$CONFIG`, and then start evaluation:
```
sh demo.sh
```


## Models
All the trained models can be found [here](https://pan.baidu.com/s/1y84hVR0RYYONGJDs8SQJAg) with fetch code `mmt9`.

![SOTA](https://github.com/ming71/RIDet/blob/RIDet-pytorch/pics/performance.png)

## Detections

![Dets](https://github.com/ming71/RIDet/blob/RIDet-pytorch/pics/DOTA.jpg)

## Notes
The implementation does not work well on the scene text datasets. Recommend to my another implementation: [RIDet-pytorch](https://github.com/ming71/RIDet/tree/RIDet-pytorch). 

## Citation

If you find our work or code useful in your research, please consider citing:

```
@article{ming2021optimization, 
	author={Ming, Qi and Miao, Lingjuan and Zhou, Zhiqiang and Yang, Xue and Dong, Yunpeng}, 
	journal={IEEE Geoscience and Remote Sensing Letters}, 
	title={Optimization for Arbitrary-Oriented Object Detection via Representation Invariance Loss}, 
	year={2021}, 
	pages={1-5}, 
	doi={10.1109/LGRS.2021.3115110}}
```

If you have any questions, please contact me via issue or [email](mq_chaser@126.com).

