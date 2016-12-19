#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utilities for the training process and logging"""

import glob
import tensorflow as tf


def build_name(args):
    """Build method name parsing args"""
    optimizer = getattr(tf.train, args.optimizer)(**args.optimizer_args)
    learning_rate = args.optimizer_args["learning_rate"]
    name = '{}_{}_lr={}_'.format(args.dataset, optimizer._name, learning_rate)

    if args.lr_decay:
        name += 'exp_lr_'
    if args.l2_penalty != 0.0:
        name += 'l2={}_'.format(args.l2_penalty)
    if args.comment != '':
        name += '{}_'.format(args.comment)

    return name.rstrip('_')


def get_models():
    """Returns the avaiable modules filename, without the .py ext"""
    return [
        model[len('models/'):-3] for model in glob.glob('models/*.py')
        if "__init__.py" not in model and "utils" not in model and
        "Autoencoder.py" not in model and "Classifier.py" not in model
    ]


def get_datasets():
    """Returns the avaiable datasets filename, without the .py ext"""
    return [
        dataset[len('inputs/'):-3] for dataset in glob.glob('inputs/*.py')
        if "__init__.py" not in dataset and "utils" not in dataset and
        "Input.py" not in dataset
    ]


def get_optimizers():
    """Returns the avaiable Tensorflow optimizer"""
    return [
        optimizer for optimizer in dir(tf.train)
        if optimizer.endswith("Optimizer")
    ]
