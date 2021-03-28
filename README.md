# Optimization for Oriented Object Detection via Representation Invariance Loss
By Qi Ming, Zhiqiang Zhou, Lingjuan Miao, Xue Yang, and Yunpeng Dong.

The repository hosts the codes for our paper `Optimization for Oriented Object Detection via Representation Invariance Loss` ([paper link](https://arxiv.org/abs/2103.11636)), based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [s2anet](https://github.com/csuhan/s2anet). 


## Introduction
To be updated.


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
All the trained models can be found [here](https://pan.baidu.com/s/1jBRHu4VaNAbqHVYH71Y47A) with fetch code `q9zc`.

## Notes
The implementation based on mmdetection does not work well on the scene text datasets. Recommend to use my another implementation: [RIDet-pytorch](https://github.com/ming71/RIDet/tree/RIDet-pytorch). 

## Citation
```
@article{ming2021optimization,
  title={Optimization for Oriented Object Detection via Representation Invariance Loss},
  author={Ming, Qi and Zhou, Zhiqiang and Miao, Lingjuan and Yang, Xue and Dong, Yunpeng},
  journal={arXiv preprint arXiv:2103.11636},
  year={2021}
}
```

