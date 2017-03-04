#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Train method and utilities"""

import math
import os
import tensorflow as tf
from .models.interfaces import Autoencoder, Classifier, Detector, Regressor
from .inputs.interfaces import InputType
from .trainers.AutoencoderTrainer import AutoencoderTrainer
from .trainers.ClassifierTrainer import ClassifierTrainer
from .trainers.DetectorTrainer import DetectorTrainer
from .trainers.RegressorTrainer import RegressorTrainer


def _build_name(args):
    # TODO: use every args, other subatpaths?
    """Build method name parsing args"""
    optimizer = args["gd"]["optimizer"](**args["gd"]["args"])
    learning_rate = args["gd"]["args"]["learning_rate"]
    optimizer_name = optimizer.get_name()
    dataset_name = args["dataset"].name
    name = "{}_{}_lr={}_".format(dataset_name, optimizer_name, learning_rate)

    if args["lr_decay"]["enabled"]:
        name += "exp_lr_"
    if args["regularizations"]["l2"]:
        name += "l2={}".format(args["regularizations"]["l2"])
    if args["comment"] != "":
        name += "{}".format(args["comment"])

    return name.rstrip("_")


def _parse_hyperparameters(hyperparams={}):
    """Check if every parameter passed in hyperparams
    is a valid hyperparameter.
    Returns:
        hyperparams: the same dictionary with default values added if optionals
    Raises:
        ValueError if hyperparams is not valid"""

    # Instantiate with default values if not specified
    args = {
        "batch_size":
        hyperparams.get("batch_size", 50),
        "epochs":
        hyperparams.get("epochs", 150),
        "gd":
        hyperparams.get("gd", {
            "optimizer": tf.train.MomentumOptimizer,
            "args": {
                "learning_rate": 1e-3,
                "momentum": 0.9,
                "use_nesterov": False
            }
        }),
        "lr_decay":
        hyperparams.get("lr_decay",
                        {"enabled": False,
                         "epochs": 25,
                         "factor": .1}),
        "regularizations":
        hyperparams.get("regularizations",
                        {"l2": 0.0,
                         "augmentation": lambda image: image})
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


def _parse_surgery(surgery={}):
    """Check if every parameter passed in surgery is valid
    for network surgery purposes.

    Returns:
        surgery: the same dictionary with defautl values added if needed
    Raises:
        ValueError if surgery values are not valid"""

    args = {
        "checkpoint_path": surgery.get("checkpoint_path", ""),
        "exclude_scopes": surgery.get("exclude_scopes", ""),
        "trainable_scopes": surgery.get("trainable_scopes", ""),
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
          hyperparameters={},
          surgery={},
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
    args = {**hyperparameters, **surgery, "force_restart": force_restart, "model": model, "dataset": dataset, "comment": comment}

    #### Training constants ####
    steps_per_epoch = math.ceil(args["dataset"].num_examples(InputType.train) /
                                args["batch_size"])

    steps = {
        "epoch": steps_per_epoch,
        "log": math.ceil(steps_per_epoch / 10),
        "max": steps_per_epoch * args["epochs"],
        "decay": steps_per_epoch * args["lr_decay"]["epochs"]
    }

    #### Model logs and checkpoint constants ####
    current_dir = os.getcwd()
    name = _build_name(args)
    log_dir = os.path.join(current_dir, "log", args["model"].name, name)
    best_dir = os.path.join(log_dir, "best")
    paths = {"current": current_dir, "log": log_dir, "best": best_dir}

    if tf.gfile.Exists(log_dir) and force_restart:
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    if not tf.gfile.Exists(best_dir):
        tf.gfile.MakeDirs(best_dir)

    if isinstance(model, Classifier):
        trainer = ClassifierTrainer()
    elif isinstance(model, Autoencoder):
        trainer = AutoencoderTrainer()
    elif isinstance(model, Regressor):
        trainer = RegressorTrainer()
    elif isinstance(model, Detector):
        trainer = DetectorTrainer()
    else:
        raise ValueError("train method not defined for this model type")

    return trainer.train(model, dataset, args, steps, paths)
