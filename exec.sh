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
