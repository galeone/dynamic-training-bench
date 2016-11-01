# Model
The base model is the one described in

- https://github.com/szagoruyko/cifar.torch
- http://torch.ch/blog/2015/07/30/cifar.html

That's a VGG-like model, whose input is a 32x32x3 image. The author claims to achieve 92.45% on CIFAR-10, using Torch.

# Preprocessing
- No conversion from RGB to YUV has been done.
- No normalization over the whole train set has been done: no means where colected.
- The images are singularly whitened (each channel is normalized).

## Data Augmentation
The only data augmentation done is the horizontal flip.

# Measurements
Different training procedures have been evaluated.
When using (mini-batch) SGD + Momentum, the following parameters has been used when no otherwise specified.

- Momentum: 0.9
- Learning rate: 1e-2
- Decay fator: 0.1
- Decay every 25 epochs
- Train for 300 epochs
- Weight decay: 5e-4

When *keep_prob decay* is specified, the following parameters has been used:

- the dropout keep probability is the *same* for every dropout layer
- the keep probability is decreased by a factor of 0.05 using the supervised parameter decay, using the validation accuracy as evaluation metric
- initial keep probability: 1.0
- precision: 1e-3
- number of measurement: 10


# Original architecture

The following tests have been made on the original architecture. This means that batch normalization layers (when presents) and dropout layers (if presents) are in the same position of the original architecture.

## Test 1

```
python train.py --model model1 --dataset cifar10 --lr_decay
```

- Dropout layers in the same position with the same keep probabilities
- No BN
- LR decay

*Best validation accuracy*: 0.829

## Test 2

```
python train.py --model model3 --dataset cifar10 --lr_decay
```

- No dropout
- BN
- LR decay

*best validation accuracy*: 0.8798

## Test 3

```
python train.py --model model2 --dataset cifar10 --lr_decay
```

- No Dropout
- No BN
- LR decay

*Best validation accuracy*: 0.8654

# Modified architecture

This architecture is equal to the original one, the only difference is the position of the dropout layers (when present): the dropout layers have been placed only at the beginning ef every block of convolutional filters with the same number of outputs (64 features, 128 fatures, ...) and after every fully connected layer.


## Test 1

```
python train.py --model model2 --dataset cifar10 --lr_decay --kp_decay
```

- Dropout: keep_prob decay
- No BN
- LR decay

*Best validation accuracy*: 0.8709


## Test 2

```
python train.py --model model2 --dataset cifar10 --kp_decay
```

- Dropout: keep_prob decay
- No BN
- No LR decay

*best validation accuracy*: 0.888

## Test 3

```
python train.py --model model3 --dataset cifar10 --kp_decay
```

- Dropout: keep_prob decay
- BN
- No LR decay

*best validation accuracy*: 0.8812

## Test 4

```
python train.py --model model3 --dataset cifar10 --kp_decay --lr_decay
```

- Dropout: keep_prob decay
- BN
- LR decay

*best validation accuracy*: 0.8842
