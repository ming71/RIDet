# Optimization for Oriented Object Detection via Representation Invariance Loss
By Qi Ming, Zhiqiang Zhou, Lingjuan Miao, Xue Yang, and Yunpeng Dong.

The repository hosts the codes for our paper `Optimization for Oriented Object Detection via Representation Invariance Loss` (paper link). 

## Notes


## Introduction
To be updated.



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
All the trained models can be found [here](https://pan.baidu.com/s/1jBRHu4VaNAbqHVYH71Y47A) with fetch code `q9zc`.

## Citation
To be updated.