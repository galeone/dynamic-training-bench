from sys import exit
import tensorflow as tf
from dtb.inputs import MNIST, Cifar10, ORLFaces
from dtb.inputs import Cifar10
from dtb.train import train
from LeNet import LeNet
from dtb.models.LeNetDropout import LeNetDropout


def main():
    model = LeNet()
    dataset = MNIST.MNIST()
    #dataset = Cifar10.Cifar10()
    #dataset = ORLFaces.ORLFaces()
    #dataset = None

    # First train LeNet on MNIST from scratch
    with tf.device("/gpu:0"):
        train(
            model,
            dataset,
            hyperparameters={
                "epochs": 1,
                "batch_size": 50,
                "regularizations": {
                    "l2":
                    1e-5,
                    "augmentation":
                    lambda image: tf.image.random_flip_left_right(image)
                },
                "lr_decay": {
                    "epochs": 25,
                    "factor": .1
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
            },
            force_restart=True)

    # Then train it again for another epoch
    with tf.device("/gpu:0"):
        train(
            model,
            dataset,
            hyperparameters={
                "epochs": 2,
                "batch_size": 50,
                "regularizations": {
                    "l2":
                    1e-5,
                    "augmentation":
                    lambda image: tf.image.random_flip_left_right(image)
                },
                "lr_decay": {
                    "epochs": 25,
                    "factor": .1
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

    # Now use the model with the highest validation accuracy reached,
    # as starting point to train the same model with a dropout layer applied
    # on the last fully connected layer
    with tf.device("/gpu:0"):
        train(
            model,
            dataset,
            hyperparameters={
                "epochs": 1,
                "batch_size": 50,
                "regularizations": {
                    "l2":
                    1e-5,
                    "augmentation":
                    lambda image: tf.image.random_flip_left_right(image)
                },
                "lr_decay": {
                    "epochs": 25,
                    "factor": .1
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
            },
            force_restart=True,
            surgery={
                "checkpoint_path": "log/LeNetDropout/MNIST_Momentum_ecc/best/",
                "exclude_scopes": "LeNet/softmax_linear",
                "trainable_scopes": "LeNet/softmax_linear"
            })
    return 0


if __name__ == "__main__":
    exit(main())
