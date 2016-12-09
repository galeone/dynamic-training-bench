#!/usr/bin/env bash

##############################################
### Test various regularization techniques ###
##############################################

# VGG: no regularization
python train_classifier.py --model VGG --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# VGG: L2 (every layer)
python train_classifier.py --model VGG --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024 --l2_penalty "5e-4"

# VGG: BN (every layer)
python train_classifier.py --model VGGBN --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# VGG: dropout (every layer)
python train_classifier.py --model VGGDropout --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# VGG: binomial dropout (every layer)
python train_classifier.py --model VGGBinomialDropout --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# VGG: direct dropout (every layer)
python train_classifier.py --model VGGDirectDropout --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# VGG: direct binomial dropout (every layer)
python train_classifier.py --model VGGDirectBinomialDropout --dataset Cifar10 --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# LeNet: no regularization
python train_classifier.py --model LeNet --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# LeNet: L2 (every layer)
python train_classifier.py --model LeNet --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024 --l2_penalty "5e-4"

# LeNet: BN (every layer)
python train_classifier.py --model LeNetBN --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# LeNet: dropout (every layer)
python train_classifier.py --model LeNetDropout --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# LeNet: binomial dropout (every layer)
python train_classifier.py --model LeNetBinomialDropout --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# LeNet: direct dropout (every layer)
python train_classifier.py --model LeNetDirectDropout --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024

# LeNet: direct binomial dropout (every layer)
python train_classifier.py --model LeNetDirectBinomialDropout --dataset MNIST --epochs 50 \
    --train_device "/gpu:1" \
    --batch_size 1024
#!/usr/bin/env bash

python train_autoencoder.py \
    --model SingleLayerCAE \
    --dataset MNIST \
    --optimizer AdamOptimizer \
    --optimizer_args '{"learning_rate": 1e-5}' \
    --train_device "/gpu:1" --batch_size 1024

python train_autoencoder.py \
    --model SingleLayerCAE \
    --dataset MNIST \
    --optimizer AdamOptimizer \
    --optimizer_args '{"learning_rate": 1e-5}' \
    --train_device "/gpu:1" \
    --l2_penalty 1e-9 --batch_size 1024


python train_autoencoder.py \
    --model SingleLayerCAE \
    --dataset Cifar10 \
    --optimizer AdamOptimizer \
    --optimizer_args '{"learning_rate": 1e-5}' \
    --train_device "/gpu:1" --batch_size 1024

python train_autoencoder.py \
    --model SingleLayerCAE \
    --dataset Cifar10 \
    --optimizer AdamOptimizer \
    --optimizer_args '{"learning_rate": 1e-5}' \
    --train_device "/gpu:1" \
    --l2_penalty 1e-9 --batch_size 1024

