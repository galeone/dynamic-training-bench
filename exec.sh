#!/usr/bin/env bash

##############################################
### Test various regularization techniques ###
##############################################

# VGG: no regularization
python train_classifier.py --model VGG --dataset Cifar10 --epochs 50

# VGG: L2 (every layer)
python train_classifier.py --model VGG --dataset Cifar10 --epochs 50 --l2_penalty "5e-4"

# VGG: BN (every layer)
python train_classifier.py --model VGGBN --dataset Cifar10 --epochs 50

# VGG: dropout (every layer)
python train_classifier.py --model VGGDropout --dataset Cifar10 --epochs 50

# VGG: binomial dropout (every layer)
python train_classifier.py --model VGGBinomialDropout --dataset Cifar10 --epochs 50

# VGG: direct dropout (every layer)
python train_classifier.py --model VGGDirectDropout --dataset Cifar10 --epochs 50

# VGG: direct binomial dropout (every layer)
python train_classifier.py --model VGGDirectBinomialDropout --dataset Cifar10 --epochs 50

# LeNet: no regularization
python train_classifier.py --model LeNet --dataset MNIST --epochs 50

# LeNet: L2 (every layer)
python train_classifier.py --model LeNet --dataset MNIST --epochs 50 --l2_penalty "5e-4"

# LeNet: BN (every layer)
python train_classifier.py --model LeNetBN --dataset MNIST --epochs 50

# LeNet: dropout (every layer)
python train_classifier.py --model LeNetDropout --dataset MNIST --epochs 50

# LeNet: binomial dropout (every layer)
python train_classifier.py --model LeNetBinomialDropout --dataset MNIST --epochs 50

# LeNet: direct dropout (every layer)
python train_classifier.py --model LeNetDirectDropout --dataset MNIST --epochs 50

# LeNet: direct binomial dropout (every layer)
python train_classifier.py --model LeNetDirectBinomialDropout --dataset MNIST --epochs 50
