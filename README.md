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
