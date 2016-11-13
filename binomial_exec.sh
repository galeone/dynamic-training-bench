#!/usr/bin/env bash

########################################################
### Test binomial dropout with other regularizations ###
########################################################

# Original model with BiDrop only
python train.py --model model7 --dataset cifar10 --lr "2e-2"

# Original model with BiDrop & learning rate decay
python train.py --model model7 --dataset cifar10 --lr_decay

# Original model with BiDrop & kp decay
python train.py --model model8 --dataset cifar10 --kp_decay

# Original model with BiDrop & BN
python train.py --model model9 --dataset cifar10 --kp_decay

# BiDrop to every layer
python train.py --model model10 --dataset cifar10 --lr "2e-2"

# BiDrop & BN to every layer
python train.py --model model11 --dataset cifar10 --lr "2e-2"

# BiDrop & l2 regularization to every layer
python train.py --model model10 --dataset cifar10 --l2_penalty "5e-4" --lr "2e-2"
