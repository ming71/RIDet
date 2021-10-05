# Optimization for Oriented Object Detection via Representation Invariance Loss

<div align=center><img width="500" height="300" src="https://github.com/ming71/RIDet/blob/RIDet-pytorch/pics/RIL.png"/></div>

By [Qi Ming](https://ming71.github.io/), Zhiqiang Zhou, Lingjuan Miao, [Xue Yang](https://yangxue0827.github.io/index.html), and Yunpeng Dong.

The repository hosts the codes for our paper `Optimization for Oriented Object Detection via Representation Invariance Loss` ([PDF](https://ieeexplore.ieee.org/document/9555916), [arxiv](https://arxiv.org/abs/2103.11636)). 


## Introduction
Arbitrary-oriented objects exist widely in remote sensing images. The mainstream rotation detectors use oriented bounding boxes (OBB) or quadrilateral bounding boxes (QBB) to represent the rotating objects. However, these methods suffer from the representation ambiguity for oriented object definition, which leads to suboptimal regression optimization and the inconsistency between the loss metric and the localization accuracy of the predictions. In this paper, we propose a Representation Invariance Loss (RIL) to optimize the bounding box regression for the rotating objects in the remote sensing images. RIL treats multiple representations of an oriented object as multiple equivalent local minima, and hence transforms bounding box regression into an adaptive matching process with these local minima. Next, the Hungarian matching algorithm is adopted to obtain the optimal regression strategy. Besides, we propose a normalized rotation loss to alleviate the weak correlation between different variables and their unbalanced loss contribution in OBB representation. Extensive experiments on remote sensing datasets show that our method achieves consistent and substantial improvement. 



## Getting Started

### Installation
```
conda create -n ridet python=3.6 -y
source activate ridet
conda install pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch
```
Build the Cython  and CUDA modules:
```
cd $ROOT/utils
sh make.sh
```
Install DOTA_devkit:
```
cd $ROOT/datasets/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
Install requirements:
```
pip install requirements.txt
pip install git+git://github.com/lehduong/torch-warmup-lr.git
```

### Inference
```
python demo.py
```

### Train
1. prepare dataset and move it into the `$ROOT` directory.
2. generate imageset files:
```
cd $ROOT/datasets
python generate_imageset.py
```
3. start training:
```
python train.py
```
### Evaluation
prepare labels, take hrsc for example:
```
cd $ROOT/datasets/evaluate
python hrsc2gt.py
```
start evaluation:
```
python eval.py
```
Note that :

- the script  needs to be executed **only once**.
- the imageset file used in `hrsc2gt.py` is generated from `generate_imageset.py`.


### Supported datasets
* DOTA
* HRSC2016
* ICDAR2013
* ICDAR2015
* UCAS-AOD
* NWPU VHR-10
* VOC2007
* MSRA-TD500


## Models
All the trained models can be found [here](https://pan.baidu.com/s/1y84hVR0RYYONGJDs8SQJAg) with fetch code `mmt9`.

![SOTA](https://github.com/ming71/RIDet/tree/RIDet-pytorch/pics/performance.png)

## Detections

![Dets](https://github.com/ming71/RIDet/blob/RIDet-pytorch/pics/DOTA.jpg)

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