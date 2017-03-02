import pprint
import tensorflow as tf
from sys import exit
from dtb.inputs import Cifar10, Cifar100
from dtb.train import train
from dtb.models.VGG import VGG
from dtb.models.VGGDropout import VGGDropout


def main():
    # PrettyPrinter to display train info
    pp = pprint.PrettyPrinter(indent=4)

    vgg = VGG()
    cifar10 = Cifar10.Cifar10()

    # Train VGG on Cifar10 for an Epoch
    with tf.device("/gpu:0"):
        info = train(
            model=vgg,
            dataset=cifar10,
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

    pp.pprint(info)

    # Then train it again for another epoch
    # Note the force_restart parameter removed.
    # epochs is the TOTAL number of epoch for the trained model
    # Thus since we trained it before for a single epoch,
    # we set "epochs": 2 in order to train it for another epoch
    with tf.device("/gpu:0"):
        info = train(
            model=vgg,
            dataset=cifar10,
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
    pp.pprint(info)

    vggInfo = info

    # TRANSFER LEARNING:
    # Use the best model trained on Cifar10, to classify Cifar 100 images.
    # Thus we train ONLY the softmax linear scope (that has 100 neurons, now),
    # keeping constant any other previosly trained layer
    cifar100 = Cifar100.Cifar100()
    with tf.device("/gpu:0"):
        transferInfo = train(
            model=vgg,
            dataset=cifar100,
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
                "checkpoint_path": vggInfo["checkpoint_path"],
                "exclude_scopes": "VGG/softmax_linear",
                "trainable_scopes": "VGG/softmax_linear"
            })

    # FINE TUNING:
    # Use the model pointed by vggInfo to fine tune the whole network
    # and tune it on Cifar100.
    # Let's retrain the whole network end-to-end, starting from the learned weights
    # Just remove the "traiable_scopes" section from the surgery  parameter
    with tf.device("/gpu:0"):
        fineTuningInfo = train(
            model=vgg,
            dataset=cifar100,
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
                "checkpoint_path": vggInfo["checkpoint_path"],
                "exclude_scopes": "VGG/softmax_linear"
            })

    # Compare the performance of Transfer learning and Fine Tuning
    print('[!] TRANSFER_LEARNING:')
    pp.pprint(transferInfo)
    print('[!] FINE TUNING:')
    pp.pprint(fineTuningInfo)
    return 0


if __name__ == "__main__":
    exit(main())
