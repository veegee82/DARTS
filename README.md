# DARTS: Differentiable Architecture Search

Paper: https://arxiv.org/pdf/1806.09055.pdf

#### Abstract
This paper addresses the scalability challenge of architecture search by formulating
the task in a differentiable manner. Unlike conventional approaches of applying evolution
or reinforcement learning over a discrete and non-differentiable search space,
our method is based on the continuous relaxation of the architecture representation,
allowing efficient search of the architecture using gradient descent. Extensive experiments
on CIFAR-10, ImageNet, Penn Treebank and WikiText-2 show that our
algorithm excels in discovering high-performance convolutional architectures for
image classification and recurrent architectures for language modeling, while being
orders of magnitude faster than state-of-the-art non-differentiable techniques.

![bad](https://raw.githubusercontent.com/quark0/darts/master/img/darts.png)

Figure 1: Relaxing of the architecture.

### Found Architecture

![bad](https://raw.githubusercontent.com/veegee82/DARTS/master/images/architecture.png)

Figure 2: Founded architecture 

#### Accuracy and Loss
![bad](https://raw.githubusercontent.com/veegee82/DARTS/master/images/acc.PNG)
![bad](https://raw.githubusercontent.com/veegee82/DARTS/master/images/loss.PNG)

Figure 3: Accuracy and loss for test-set(left) and training-set(right)

## Installation tf_base package
1. Clone the repository
```
$ git clone https://github.com/veegee82/tf_base.git
```
2. Go to folder
```
$ cd tf_base
```
3. Install with pip3
``` 
$ pip3 install -e .
```

## Install DARTS package

1. Clone the repository
```
$ git clone https://github.com/veegee82/DARTS.git
```
2. Go to folder
```
$ cd DARTS
```
3. Install with pip3
```
$ pip3 install -e .
```

## Usage-Example

1. Training
```
$ python pipeline_trainer.py --dataset "../Data/"
```

2. Inferencing
```
$ python pipeline_inferencer.py --dataset "../Data/" --model_dir "Nails" 
```
