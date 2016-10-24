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

Different models have been evaluated, each one has been trained using Gradient Descent with momentum and learning rate decay.

- Momentum: 0.9
- Learning rate: 1e-2
- Decay fator: 0.1
- Decay every 25 epochs
- Train for 300 epochs

No. | L2 | Dropout | BN | Accuracy
--- | --- | --- | --- | ---
1   | 5e-4 | see blog post | no | 0.829
