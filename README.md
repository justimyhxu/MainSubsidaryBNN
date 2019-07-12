# A-Main-Subsidiary-Network-Framework-For-Simplifying-Binary-Neural-Network
This repository is for the paper 'A Main Subsidiary Network Framework For Simplifying Binary Neural Network' (in CVPR 2019) with Pytorch v0.40+.

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

You will need at least 1x 12GB GPUs to reproduce our result. As a layer-wise training method, it requires  N(N is the numbers of binary conv layers ) times  training to get the final pruned model. You can definitly implement the whole process into a for loop (especially for very deep networks like ResNet-101). However, to better illustrate the pipelien of our method, we explicitly show how to prune each layer here.

1. Prepare a normal binary model

    ~~~
    python train.py --main --lr 1e-3
    ~~~
    
Do 2,3 until the last layer was pruned. (**N** starts from 1.)

2. Prune the **N**-th layer (Training of subsidary component)

    ~~~
    python train.py --layer N --lr 1e-3 --pretrained $MODEL_PATH$ (the main network whose 1<->(N-1) layers are pruned) 
    ~~~ 
3. Fintune model (Training of main network)

   ~~~
   python train.py --main --layer N --lr 1e-4 --pretrained $MODEL_PATH (the resulted subsidary component from step 2)$
   ~~~

>Here, alpha is a hyper-parameter to control pruning ratio, during our experiments we set it in [1e-7,1e-8,1e-9], generally we use 1e-8.

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


