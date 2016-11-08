# Model
The base model is the one described in

- https://github.com/szagoruyko/cifar.torch
- http://torch.ch/blog/2015/07/30/cifar.html

That's a VGG-like model, whose input is a 32x32x3 image. The author claims to achieve 92.45% on CIFAR-10, using Torch.

# Preprocessing
- No conversion from RGB to YUV has been done.
- No normalization over the whole train set has been done: no means where collected.
- The images are singularly whitened (each channel is normalized).

## Data Augmentation
The only data augmentation done is the horizontal flip.

# Measurements
Different training procedures have been evaluated.
When using (mini-batch) SGD + Momentum, the following parameters has been used when no otherwise specified.

- Momentum: 0.9
- Learning rate: 1e-2
- Decay factor: 0.1
- Decay every 25 epochs
- Train for 300 epochs
- Weight decay: 5e-4

When *KP decay* is specified, the following parameters has been used:

- the dropout keep probability is the *same* for every dropout layer
- the keep probability is decreased by a factor of 0.05 using the supervised parameter decay, using the validation accuracy as evaluation metric
- initial keep probability: 1.0
- precision: 1e-3
- number of measurement: 10


# Architecture 1

The following tests have been made on the original architecture. This means that batch normalization layers (when presents) and dropout layers (if presents) are in the same position of the original architecture.

Test are divided in two parts: single regularization methods and multiple regularization methods.

## Single regularizer

### Test 1: no regularization at all

```
python train.py --model model2 --dataset cifar10
```

- No dropout
- No BN
- No LR decay
- No L2


*Best validation accuracy*: 0.8572

### Test 2: dropout only with fixed values

```
python train.py --model model1 --dataset cifar1
```

- Dropout with fixed values
- No BN
- No LR decay
- No L2


*Best validation accuracy*: 0.8875

*Notes*: stuck in a local minimum for a great number of epochs. Once the SGD + Momemntum has been able to escape from it, the validations started to increase and the loss to decrease.

### Test 3: batch normalization only

```
python train.py --model model3 --dataset cifar10
```

- No Dropout
- BN
- No LR decay
- No L2


*Best validation accuracy*: 0.8731


### Test 4: L2 only

```
python train.py --model model2 --dataset cifar10 --l2_penalty "5e-4"
```

- No dropout
- No BN
- No LR decay
- L2


*Best validation accuracy*: 0.8693

### Test 5: LR decay only

```
python train.py --model model2 --dataset cifar10 --lr_decay
```

- No dropout
- No BN
- LR decay
- No L2


*Best validation accuracy*: 0.8497

### Test 6: KP decay only

```
python train.py --model model6 --dataset cifar10 --kp_decay
```

- No dropout
- No BN
- No LR decay
- No L2
- KP decay

*Best validation accuracy*: 0.875

*Notes*: once the keep prob reached 0.55, the model started to worsen its performance until it diverged.

## Combined regularizers

### Test 1: dropout with fixed value + L2 

```
python train.py --model model1 --dataset cifar10 --lr_decay
```

- Dropout layers in the same position with the same keep probabilities
- No BN
- LR decay
- L2


*Best validation accuracy*: 0.829

*Notes*: very noisy training process. Training accuracy do not reach 1.

### Test 2: LR decay + BN + L2

```
python train.py --model model3 --dataset cifar10 --lr_decay
```

- No dropout
- BN
- LR decay
- L2


*Best validation accuracy*: 0.8798

*Notes*: Training accuracy and validation accuracy with a fixed gap of about 0.13. Only the first learning rate decay has been useful; 0.79 -> 0.87 accuracy.

### Test 3: LR decay + L2

```
python train.py --model model2 --dataset cifar10 --lr_decay
```

- No Dropout
- No BN
- LR decay
- L2


*Best validation accuracy*: 0.8654

*Notes*: fixed gap between training and validation accuracy (0.14). Useful only the first learning rate decay.

## Architecture 1: summary

Dropout      | BN  | LR decay | L2  | VA
-------------| --- | -------- | --- | ---
             |     |          |     | 0.8572
Fixed values |     |          |     | 0.8875
             | yes |          |     | 0.8731
             |     |          | yes | 0.8693
             |     | yes      |     | 0.8497
KP decay     |     |          |     | 0.875
Fixed values |     |          | yes | 0.829
             | yes | yes      | yes | 0.8798
             |     | yes      | yes | 0.8654


# Architecture 2

This architecture is equal to the original one, the only difference is the position of the dropout layers (when present): the dropout layers have been placed only at the beginning of every block of convolutional filters with the same number of outputs (64 features, 128 features, ...) and after every fully connected layer.

## Combined regularizers

### Test 1: KP decay + LR decay + L2

```
python train.py --model model2 --dataset cifar10 --lr_decay --kp_decay
```

- Dropout: KP decay
- No BN
- LR decay
- L2

*Best validation accuracy*: 0.8709

*Notes*: overfits training data. Useful only first LR decay 0.82 -> 0.86. Loss incredibly noisy mixing LR decay and kp decay.

### Test 2: KP decay + L2

```
python train.py --model model2 --dataset cifar10 --kp_decay
```

- Dropout: KP decay
- No BN
- No LR decay
- L2

*Best validation accuracy*: 0.888

*Notes*: less noisy training process. Training validation accuracy is not fixed at 1. Validation accuracy is a bit noisy but increases.

*Hypothesis*: start with a lower learning rate to reduce the fluctuations -> 1e-3 -> tested: it simply lowers the training speed without any other befit.


### Test 3: KP decay + BN + L2

```
python train.py --model model3 --dataset cifar10 --kp_decay
```

- Dropout: KP decay
- BN
- No LR decay
- L2

*Best validation accuracy*: 0.8812

*Notes*: w.r.t. Architecture 2, Test 2, the training process is noisier.

*Hypothesis*: lower the learning rate to 1e-3: tested -> it simply lowers the training speed without any other befit.

### Test 4: KP decay + BN + LR decay + L2

```
python train.py --model model3 --dataset cifar10 --kp_decay --lr_decay
```

- Dropout: KP decay
- BN
- LR decay
- L2

*Best validation accuracy*: 0.8842

*Notes*:  first learning rate decay increases the accuracy instantly. This support the hypothesis of starting from a lower learning rate when using keep prob decay (with and without batch norm. -> tested: it simply lowers the training speed without any other befit.

## Architecture 2: summary

Dropout      | BN  | LR decay | L2  | VA
-------------| --- | -------- | --- | ---
KP decay     |     | yes      | yes | 0.8709
KP decay     |     |          | yes | 0.888
KP decay     | yes |          | yes | 0.8812
KP decay     | yes | yes      | yes | 0.8842


# Architecture 3

Like architecture 2, but with dropout layer after every convolutional/fc layer.

### Test 1: KP decay + L2

```
python train.py --model model4 --dataset cifar10 --kp_decay
```

- Dropout: KP decay
- No BN
- No LR decay
- L2

*Best validation accuracy*: 0.8897

*Notes*: extremely noisy training process. When keep prob goes under 0.5 the validation accuracy start decreasing and the model underfits the training too.

### Test 2: KP decay + BN + L2

```
python train.py --model model5 --dataset cifar10 --kp_decay
```

- Dropout: KP decay
- BN
- No LR decay
- L2

*Best validation accuracy*: 0.8409

*Notes*: validation accuracy decreases after kp goes under 1. *Extremely* noisy training process. When kp decreases, underfits too.

## Architecture 3: summary

Dropout      | BN  | LR decay | L2  | VA
-------------| --- | -------- | --- | ---
KP decay     |     |          | yes | 0.8897
KP decay     | yes |          | yes | 0.8409

