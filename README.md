# Model
The base model is the one described in

- https://github.com/szagoruyko/cifar.torch
- http://torch.ch/blog/2015/07/30/cifar.html

That's a VGG-like model, whose input is a 32x32x3 image. The author claims to achieve 92.45% on CIFAR-10, using Torch.

# Preprocessing
Instead of converting input images from RGB to YUV colorspace, we use RGB.

The images are singularly whitened (each channel is normalized).

No normalization over the whole train set has been done: no means where colected.

# Data Augmentation
The only data augmentation done is the horizontal flip.

# Measurements

Different models have been evaluated.
When using (mini-batch) SGD + Momentum, the following parameters has been used when no otherwise specified.

- Momentum: 0.9
- Learning rate: 1e-2
- Decay fator: 0.1
- Decay every 25 epochs
- Train for 300 epochs
- Weight decay: 5e-4

## Test 1

- Dropout layers in the same position with the same keep probabilities
- No BN

*Best validation accuracy*: 0.829

## Test 2

- No Dropout
- No BN

*Best validation accuracy*: 0.859

## Test 3

Dropout layers at the beginning of every block of convolutional filters with the same number of outputs (64 features, 128 fatures, ...) and FC layers. *Same* keep probability for every layer.

### keep_prob decay and learning rate decay

- number of measurement: 10
- precision: 1e-3
- initial keep prob: 1.0
- final keep_prob: 0.4
- decay: 0.05

*best validation accuracy*: 0.8709

### keep_prob decay only

- number of measurement: 10
- precision: 1e-3
- initial keep prob: 1.0
- final keep_prob: 0.4
- decay: 0.05

*best validation accuracy*: 0.888
