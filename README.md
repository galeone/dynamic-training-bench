Dynamic Training Bench: DyTB
===========================

Stop wasting your time rewriting the training, evaluation & visualization procedures for your ML model: let DyTB do the work for you!

DyTB is compatible with: **Tensorflow 1.x & Python 3.x**

# Features

1. Dramatically easy to use
2. Object Oriented: models and inputs are interfaces to implement
3. End-to-end training of ML models
4. Fine tuning
5. Transfer learning
6. Easy model comparison
7. Metrics visualization
8. Easy statistics
9. Hyperparameters oriented: change hyperparameters to see how they affect the performance
10. Automatic checkpoint save of the best model with respect to a metric
11. Usable as a library or a CLI tool

---

# Getting started: python library

**TL;DR**: `pip install dytb` + [python-notebook with a complete example](examples/VGG-Cifar10-100-TransferLearning-FineTuning.ipynb).

The standard workflow is extremely simple:

1. Define or pick a predefined model
2. Define or pick a predefined dataset
3. Train!

## Define or pick a predefined Model

DyTB comes with some common ML model, like LeNet & VGG, if you want to test how these models perform when trained on different datasets and/or with different hyperparameters, just use it.

Instead, if you want to define your own model just implement one of the [available interfaces](dytb/models/interfaces.py), depending on ML model you want to implement. The available interfaces are:

1. Classifier
2. Autoencoder
3. Regressor
4. Detector

It's recommended, but not strictly required, to use the wrappers built around the Tensorflow methods to define the model: these wrappers creates log and visualizations for you.
Wrappers are documented and intuitive: you can find it in the [dytb/models/utils.py](dytb/models/utils.py) file.

DyTB provides different models that can be used alone or can be used as examples of correct implementations.
Every model in the [dytb/models/predefined/](dytb/models/predefined/) folder is a valid example.

In general, the model definition is just the implementation of 2 methods:

1. `get` is which implementing the model itself
2. `loss` in which implementing the loss function

It's strictly required to return the parameters that the method documentation requires to, even if they're unused by your model.

E.g.: even if you never use a `is_training_` boolean placeholder in your model definition, define it and return it anyway.

## Define or pick a predefined Input

DyTB comes with some common ML benchmark, like Cifar10, Cifar100 & MNIST, you can use it to train and measure the performances of your model or you can define your own input source implementing the Input interface that you can find here:

1. [dytb/inputs/interfaces.py](dytb/inputs/interfaces.py)

The interface implementation should follow these points:

1. Implement the `__init__` method: this method must download the dataset and apply the desired transformations to its elements. There are some utility functions defined in the [`inputs/utils.py`](inputs/utils.py) file that can be used.
This method is executed as first operation when the dataset object is created, therefore is recommended to cache the results.
2. Implement the `num_classes` method: this method must return the number of classes of the dataset. If your dataset has no labels, just return 0.
3. Implement the `num_examples(input_type)` method: this method accepts an `InputType` enumeration, defined in `inputs/utils.py`.
This enumeration has 3 possible values: `InputType.train`, `InputType.validation`, `InputType.test`. As obvious, the method must return the number of examples for every possible value of this enumeration.
4. Implement the `inputs` method. The `inputs` method is a general method that should return the real values of the dataset, related to the `InputType` passed, without any augmentation. The augmentations are defined at training time.

**Note**: `inputs` must return a Tensorflow queue of `value, label` pairs.

The better way to understand how to build the input source is to look at the examples in the [dytb/inputs/predefined/](dytb/inputs/predefined/) folder.
A small and working example that can be worth looking is Cifar10: [dytb/inputs/predefined/Cifar10.py](dytb/inputs/predefined/Cifar10.py).

## Train

Train measuring predefined metrics it's extremely easy, let's see a complete example:

```python
import pprint
import tensorflow as tf
from dytb.inputs import Cifar10
from dytb.train import train
from dytb.models.VGG import VGG

# Instantiate the model
vgg = VGG()

# Instantiate the CIFAR-10 input source
cifar10 = Cifar10.Cifar10()

# 1: Train VGG on Cifar10 for 50 epochs
# Place the train process on GPU:0
device = '/gpu:0'
with tf.device(device):
    info = train(
        model=vgg,
        dataset=cifar10,
        hyperparameters={
            "epochs": 50,
            "batch_size": 50,
            "regularizations": {
                "l2": 1e-5,
                "augmentation": {
                    "name": "FlipLR",
                    "fn": tf.image.random_flip_left_right
                }
            },
            "gd": {
                "optimizer": tf.train.AdamOptimizer,
                "args": {
                    "learning_rate": 1e-3,
                    "beta1": 0.9,
                    "beta2": 0.99,
                    "epsilon": 1e-8
                }
            }
        })
```

Finish!

At the end of the training process `info` will contain some useful information, let's (pretty) print them:

```python
pprint.pprint(info, indent=4)
```

```
{   'args': {   'batch_size': 50,
                'checkpoint_path': '',
                'comment': '',
                'dataset': <dytb.inputs.Cifar10.Cifar10 object at 0x7f896c19a1d0>,
                'epochs': 2,
                'exclude_scopes': '',
                'force_restart': False,
                'gd': {   'args': {   'beta1': 0.9,
                                      'beta2': 0.99,
                                      'epsilon': 1e-08,
                                      'learning_rate': 0.001},
                          'optimizer': <class 'tensorflow.python.training.adam.AdamOptimizer'>},
                'lr_decay': {'enabled': False, 'epochs': 25, 'factor': 0.1},
                'model': <dytb.models.VGG.VGG object at 0x7f896c19a128>,
                'regularizations': {   'augmentation': <function random_flip_left_right at 0x7f89109cb0d0>,
                                       'l2': 1e-05},
                'trainable_scopes': ''},
    'paths': {   'best': '/mnt/data/pgaleone/dytb_work/examples/log/VGG/CIFAR-10_Adam_l2=1e-05_fliplr/best',
                 'current': '/mnt/data/pgaleone/dytb_work/examples',
                 'log': '/mnt/data/pgaleone/dytb_work/examples/log/VGG/CIFAR-10_Adam_l2=1e-05_fliplr'},
    'stats': {   'dataset': 'CIFAR-10',
                 'model': 'VGG',
                 'test': 0.55899998381733895,
                 'train': 0.5740799830555916,
                 'validation': 0.55899998381733895},
    'steps': {'decay': 25000, 'epoch': 1000, 'log': 100, 'max': 2000}}
```

---

Here you can see a complete example of training, continue an interrupted training, fine tuning & transfer learning: [python-notebook with a complete example](examples/VGG-Cifar10-100-TransferLearning-FineTuning.ipynb).

# Getting started: CLI

The only prerequisite is to install DyTB via pip.

```
pip install --upgrade dytb
```

DyTB adds to your $PATH two executables: `dytb_train` and `dytb_evaluate`.

The CLI workflow is the same as the library one, with 2 differences:

## 1. Interface implementations

If you define your own input source / model, it must be placed into the appropriate folder:

- For models: [scripts/models/](scripts/models)
- For inputs: [scripts/inputs/](scripts/inputs)

**Rule**: the class name must be equal to the file name. E.g.: `class LeNet` into `LeNet.py` file.

If you want to use a predefined input/model you don't need to do anything.

## 2. Train via CLI

Every single hyperparameter (except for the augmentations) definable in the Python version, can be passed as CLI argument to the `dytb_train` script.

A single model can be trained using various hyper-parameters, such as the learning rate, the weight decay penalty applied, the exponential learning rate decay, the optimizer and its parameters, ...

DyTB allows training a model with different hyper-parameter and automatically it logs every training process allowing the developer to visually compare them.

Moreover, if a training process is interrupted, it automatically resumes it from the last saved training step.

## Example

```
# LeNet: no regularization
dytb_train --model LeNet --dataset MNIST

# LeNet: L2 regularization with value 1e-5
dytb_train --model LeNet --dataset MNIST --l2_penalty 1e-5

# LeNet: L2 regularization with value 1e-2
dytb_train --model LeNet --dataset MNIST --l2_penalty 1e-2

# LeNet: L2 regularization with value 1e-2, initial learning rate of 1e-4
# The default optimization algorithm is MomentumOptimizer, so we can change the momentum value
# The optimizer parameters are passed as a json string
dytb_train --model LeNet --dataset MNIST --l2_penalty 1e-2 \
    --optimizer_args '{"learning_rate": 1e-4, "momentum": 0.5}'

# If, for some reason, we interrupt this training process, rerunning the same command
# will restart the training process from the last saved training step.
# If we want to delete every saved model and log, we can pass the --restart flag
dytb_train --model LeNet --dataset MNIST --l2_penalty \
    --optimizer_args '{"learning_rate": 1e-4, "momentum": 0.5}' --restart
```

The commands above will create 4 different models. Every model has it's own log folder that shares the same root folder.

In particular, in the `log` folder there'll be a `LeNet` folder and within this folder, there'll be other 4 folders, each one with a name that contains the hyper-parameters previously defined.
This allows visualizing in the same graphs, using Tensorboard, the 4 models and easily understand which one performs better.

No matter what interface has been implemented, the script to run is **always** `train.py`: it's capable of identifying the type of the model and use the right training procedure.

A complete list of the available tunable parameters can be obtained running `dytb_train --help` (`dytb_train --help`).

For reference, a part of the output of `dytb_train --help`:

```
usage: train.py [-h] --model --dataset
  -h, --help            show this help message and exit
  --model {<list of models in the models/ folder, without the .py suffix>}
  --dataset {<list of inputs in the inputs/folder, without the .py suffix}
  --batch_size BATCH_SIZE
  --restart             restart the training process DELETING the old
                        checkpoint files
  --lr_decay            enable the learning rate decay
  --lr_decay_epochs LR_DECAY_EPOCHS
                        decay the learning rate every lr_decay_epochs epochs
  --lr_decay_factor LR_DECAY_FACTOR
                        decay of lr_decay_factor the initial learning rate
                        after lr_decay_epochs epochs
  --l2_penalty L2_PENALTY
                        L2 penalty term to apply ad the trained parameters
  --optimizer {<list of tensorflow available optimizers>}
                        the optimizer to use
  --optimizer_args OPTIMIZER_ARGS
                        the optimizer parameters
  --epochs EPOCHS       number of epochs to train the model
  --train_device TRAIN_DEVICE
                        the device on which place the the model during the
                        trining phase
  --comment COMMENT     comment string to preprend to the model name
  --exclude_scopes EXCLUDE_SCOPES
                        comma separated list of scopes of variables to exclude
                        from the checkpoint restoring.
  --checkpoint_path CHECKPOINT_PATH
                        the path to a checkpoint from which load the model
```

# Best models & results

No matter if the CLI or the library version is used: DyTB saves for you in the log folder of every model the "best" model with respect to the default metric used for the trained model.

For example, for the `LeNet` model created with the first command in the previous script, the following directory structure is created:

```
log/LeNet/
|---MNIST_Momentum
|-----best
|-----train
|-----validation
```

`train` and `validation` folders contain the logs, used by Tensorboard to display in the same graphs train and validation metrics.

The `best` folder contains one single checkpoint file that is the model with the highest quality obtained during the training phase.

This model is used at the end of the training process to evaluate the model performance.

Moreover, is possible to run the evaluation of any checkpoint file (in the `log/<MODEL>` folder or in the `log/<MODEL>/best` folder) using the `dytb_evaluate` script.

For example:

```
# Evaluate the validation accuracy
dytb_evaluate --model LeNet \
              --dataset MNIST \
              --checkpoint_path log/LeNet/MNIST_Momentum/
# outputs something like: validation accuracy = 0.993

# Evaluate the test accuracy
dytb_evaluate --model LeNet \
              --dataset MNIST \
              --checkpoint_path log/LeNet/MNIST_Momentum/ \
              --test
# outputs something like: test accuracy = 0.993
```

# Fine Tuning & network surgery

A trained model can be used to build a new model exploiting the learned parameters: this helps to speed up the learning process of new models.

DyTB allows to restore a model from its checkpoint file, remove some layer that's not necessary for the new model, and add new layers to train.

For example, a VGG model trained on the Cifar10 dataset, can be used to train a VGG model but on the Cifar100 dataset.

The examples are for the CLI version, **but the same parameters can be used in the Python library**.

```
dytb_train
    --model VGG \
    --dataset Cifar100 \
    --checkpoint_path log/VGG/Cifar10_Momentum/best/ \
    --exclude_scopes softmax_linear
```

This training process loads the "best" VGG model weights trained on Cifar10 from the `checkpoint_path`, then the weights are used to initialize the VGG model (so the VGG model must be compatible, at least for the non excluded scopes, to the loaded model) except for the layers under the `excluded_scopes` list.

Then the `softmax_linear` layers are replaced with the ones defined in the `VGG` model, that when trained on Cifar100 adapt themself to output 100 classes instead of 10.

So the above command starts a new training from the pre-trained model and trains the new output layer (with 100 outputs) that the VGG model defines, refining every other weights imported.

If you don't want to train the imported weights, you have to point out which scopes to train, using `trainable_scopes`:

```
dytb_train \
    --model VGG \
    --dataset Cifar100 \
    --checkpoint_path log/VGG/Cifar10_Momentum/best/ \
    --exclude_scopes softmax_linear \
    --trainable_scopes softmax_linear
```
With the above command your instructing DyTB to exclude the `softmax_linear` scope from the checkpoint_file and to train only the scope named `softmax_linear` in the new defined model.

# Data visualization

Running tensorboard

```
tensorboard --logdir log/<MODEL>
```

It's possible to visualize the trend of the loss, the validation measures, the input values and so on.
To see some of the produced output, have a look at the implementation of the Convolutional Autoencoder, described here: https://pgaleone.eu/neural-networks/deep-learning/2016/12/13/convolutional-autoencoders-in-tensorflow/#visualization

