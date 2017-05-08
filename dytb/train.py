#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Train method and utilities"""

import os
import tensorflow as tf
from .inputs.interfaces import InputType
from .trainer.Trainer import Trainer


def _build_name(args, dataset):
    """Build method name parsing args.
    Args:
        args: the training parameter
        dataset: the dataset object
    Returns:
        name: the ID for the current training process"""
    optimizer = args["gd"]["optimizer"](**args["gd"]["args"])
    name = "{}_{}_".format(dataset.name, optimizer.get_name())

    if args["lr_decay"]["enabled"]:
        name += "lr_decay_"
    if args["regularizations"]["l2"]:
        name += "l2={}_".format(args["regularizations"]["l2"])
    if args["regularizations"]["augmentation"]["name"].lower() != "identity":
        name += "{}_".format(
            args["regularizations"]["augmentation"]["name"].lower())
    if args["comment"] != "":
        name += "{}_".format(args["comment"])

    return name.rstrip("_")


def _parse_hyperparameters(hyperparams=None):
    """Check if every parameter passed in hyperparams
    is a valid hyperparameter.
    Returns:
        hyperparams: the same dictionary with default values added if optionals
    Raises:
        ValueError if hyperparams is not valid
    """

    if hyperparams is None:
        hyperparams = {}

    # Instantiate with default values if not specified
    args = {
        # The size of the trainign batch
        "batch_size":
        hyperparams.get("batch_size", 128),
        # The number of epochs to train
        # where an epoch is the training set cardinality * the augmentation factor
        "epochs":
        hyperparams.get("epochs", 150),
        # Gradient descent parameters
        "gd":
        hyperparams.get(
            "gd",
            {
                # The optimizer to use
                "optimizer": tf.train.MomentumOptimizer,
                # The arguments of the optimizer
                "args": {
                    "learning_rate": 1e-3,
                    "momentum": 0.9,
                    "use_nesterov": False
                }
            }),
        # The learning rate decay
        "lr_decay":
        hyperparams.get("lr_decay",
                        {"enabled": False,
                         "epochs": 25,
                         "factor": .1}),
        # The regularization to apply
        "regularizations":
        hyperparams.get(
            "regularizations",
            {
                # L2 on the model weights
                "l2": 0.0,
                # The augmentation on the input data: online augmentation
                "augmentation": {
                    # The name of the augmentation: identity disables the augmentations
                    "name": "identity",
                    # The function of the augmentation: fn(x) where x is the orignnal sample
                    "fn": lambda x: x,
                    # The multiplicative factor of the training set: online data augmentation
                    # can generate a potentially infinite number of training samples.
                    # However, the generated samples starts to look "similar" after
                    # being generated for a lot of times.
                    # What we do applying augmentations is to pick samples from the input
                    # distrubution.
                    # If we have enough samples (in a single epoch), we have sampled the
                    # distribution densely enough that the next epoch, altough the samples
                    # are still online generated, will look similar to the previous one.

                    # In short, this is a multiplicative factor that changes the effective
                    # training set size:
                    # 1 means no augmentation.
                    # A rule of thumb is to set this value to a power of 10.
                    "factor": 1,
                }
            }),
        # seed is the graph level and op level seed.
        # None means that random seed is used.
        # Otherwise the specified value is used.
        "seed":
        hyperparams.get("seed", None),
    }

    # Check numeric fields
    if args["epochs"] <= 0:
        raise ValueError("epochs <= 0")
    if args["batch_size"] <= 0:
        raise ValueError("batch_size <= 0")
    # The other fields will be used at runtime.
    # If they're wrong, the training process can't start
    # and tensorflow will raise errors
    return args


def _parse_surgery(surgery=None):
    """Check if every parameter passed in surgery is valid
    for network surgery purposes.

    Returns:
        surgery: the same dictionary with defautl values added if needed
    Raises:
        ValueError if surgery values are not valid
    """
    if surgery is None:
        surgery = {}

    args = {
        "checkpoint_path": surgery.get("checkpoint_path", ""),
        "exclude_scopes": surgery.get("exclude_scopes", None),
        "trainable_scopes": surgery.get("trainable_scopes", None),
    }

    if args["checkpoint_path"] != "":
        if not tf.train.latest_checkpoint(args["checkpoint_path"]):
            raise ValueError("Invalid {}".format(args["checkpoint_path"]))
    # The other fields will be used at runtime.
    # If they're wrong, the training process can't start
    # and tensorflow will raise errors
    return args


def train(model,
          dataset,
          hyperparameters=None,
          surgery=None,
          force_restart=False,
          comment=""):
    """Train the model using the provided dataset and the specifiied hyperparameters.
    Args:
        model: instance of a model interface
        dataset: instance of the Input interface
        hyperparameters: dictionary of the hyperparameter to use to train the model
        surgery: dictionary of options related to the network surgery, fine tuning and transfer
                 learning
        force_restart: boolean, indicates if restart the train from 0 removing the old model
                       or continue the training.
        comment: string to append at the log dir name
    Returns:
        info dict containing the information of the trained model
    """
    hyperparameters = _parse_hyperparameters(hyperparameters)
    surgery = _parse_surgery(surgery)
    args = {
        **hyperparameters,
        **surgery,
        "force_restart": force_restart,
        "model": model,
        "dataset": dataset,
        "comment": comment}

    name = _build_name(args, dataset)

    #### Training constants ####
    float_steps_per_epoch = dataset.num_examples(InputType.train) * args[
        "regularizations"]["augmentation"]["factor"] / args["batch_size"]
    steps_per_epoch = 1 if float_steps_per_epoch < 1. else round(
        float_steps_per_epoch)

    steps = {
        "epoch": steps_per_epoch,
        "log": 1 if steps_per_epoch < 10 else steps_per_epoch // 10,
        "max": int(float_steps_per_epoch * args["epochs"]),
        "decay": int(float_steps_per_epoch * args["lr_decay"]["epochs"]),
    }

    #### Model logs and checkpoint constants ####
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, "log", args["model"].name, name)
    best_dir = os.path.join(log_dir, "best")
    paths = {"current": current_dir, "log": log_dir, "best": best_dir}

    if tf.gfile.Exists(log_dir) and force_restart:
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    if not tf.gfile.Exists(best_dir):
        tf.gfile.MakeDirs(best_dir)

    if args["regularizations"]["augmentation"]["factor"] != 1:
        print("Original training set size {}. Augmented training set size: {}".
              format(
                  dataset.num_examples(InputType.train), args["regularizations"]
                  ["augmentation"]["factor"] * dataset.num_examples(
                      InputType.train)))
    return Trainer(model, dataset, args, steps, paths).train()
