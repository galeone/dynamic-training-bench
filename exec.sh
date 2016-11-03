#!/usr/bin/env bash

############################################################
### Test single regularization (and train) methods only  ###
############################################################

# no regularization at all
python train.py --model model2 --dataset cifar10

# fixed & different dropout values only
# model1 have hardcoded dropout values, no flags required
python train.py --model model1 --dataset cifar10

# batch normalization only
# model 3 have batch norm layer before every layer
python train.py --model model3 --dataset cifar10

# l2 regularization only
python train.py --model model2 --dataset cifar10 --l2_penalty "5e-4"

# learning rate decay only
python train.py --model model2 --dataset cifar10 --lr_decay

# keep prob decay only
python train.py --model model2 --dataset cifar10 --kp_decay
