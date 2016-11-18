#!/usr/bin/env bash

########################################################
### Test binomial dropout with other regularizations ###
########################################################

# VGG: no regularization
python train.py --model vgg --dataset cifar10

# VGG: L2 (every layer)
python train.py --model vgg --dataset cifar10 --l2_penalty "5e-4"

# VGG: BN (every layer)
python train.py --model vgg_bn --dataset cifar10

# VGG: dropout (every layer)
python train.py --model vgg_dropout --dataset cifar10

# VGG: binomial dropout (every layer)
python train.py --model vgg_binomial_dropout --dataset cifar10

# VGG: direct dropout (every layer)
python train.py --model vgg_direct_dropout --dataset cifar10

# VGG: direct binomial dropout (every layer)
python train.py --model vgg_direct_binomial_dropout --dataset cifar10

# LeNet: no regularization
python train.py --model lenet --dataset mnist

# LeNet: L2 (every layer)
python train.py --model lenet --dataset mnist --l2_penalty "5e-4"

# LeNet: BN (every layer)
python train.py --model lenet_bn --dataset mnist

# LeNet: dropout (every layer)
python train.py --model lenet_dropout --dataset mnist

# LeNet: binomial dropout (every layer)
python train.py --model lenet_binomial_dropout --dataset mnist

# LeNet: direct dropout (every layer)
python train.py --model lenet_direct_dropout --dataset mnist

# LeNet: direct binomial dropout (every layer)
python train.py --model lenet_direct_binomial_dropout --dataset mnist
