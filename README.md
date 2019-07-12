# A-Main-Subsidiary-Network-Framework-For-Simplifying-Binary-Neural-Network
This repository is for the paper 'A Main Subsidiary Network Framework For Simplifying Binary Neural Network' (To appear in CVPR 2019) with Pytorch v0.40+.

## Introduction

## Installation
The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v0.4.1. After install Anaconda:

1. Clone this repo:

    ~~~ 
    git clone https://github.com/justimyhxu/MainSubsidaryBNN.git
    ~~~


2. Create an Pythln environment 
    ~~~
    pip install -r requirements.txt
    ~~~

## Training 

You will need 1x 12GB GPUs to reproduce our training. Because it is a layer-wise training process, it means we should training  N(N is the numbers of binary conv layers ) times to get a pruned model. 

1. Prepare a normal binary model

    ~~~
    python train.py --main --lr 1e-3
    ~~~

2. Pruned the first I layer(Training subsidary component)

    ~~~
    python train.py --layer I --lr 1e-3 --pretrained $MODEL_PATH$(first I-1 layer main model Path) 
    ~~~ 
3. Fintune model (Training Main network)

   ~~~
   python train.py --main --layer I --lr 1e-4 --pretrained $MODEL_PATH(first I layer subsidary model path)$
   ~~~
Do 2,3 until the last layer was pruned.

## Citation
If you use our code/model/data, please cite our paper:
```
@article{xu2018main,
  title={A Main/Subsidiary Network Framework for Simplifying Binary Neural Network},
  author={Xu, Yinghao and Dong, Xin and Li, Yudian and Su, Hao},
  journal={arXiv preprint arXiv:1812.04210},
  year={2018}
}
```


