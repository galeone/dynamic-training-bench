#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utilities for the training process and logging"""

import glob
import argparse
import json
import importlib
import pprint
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


def parse_args(description="Train the model"):
    """Parser the CLI arguments and returns (see Returns)
    Args:
        description: The description to show when the help is displayed
    Returns:
        args: args object
        model name: string representing the model name
        model: model object instantiated
        dataset: input object instantiated
        optimizer: optimizer object instantiated"""
    # CLI arguments
    parser = argparse.ArgumentParser(description=description)

    # Required arguments
    parser.add_argument('--model', required=True, choices=get_models())
    parser.add_argument('--dataset', required=True, choices=get_datasets())

    # Restart train or continue
    parser.add_argument('--restart', action='store_true')

    # Learning rate decay arguments
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_epochs', type=int, default=25)
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)

    # L2 regularization arguments
    parser.add_argument('--l2_penalty', type=float, default=0.0)

    # Optimization arguments
    parser.add_argument(
        '--optimizer', choices=get_optimizers(), default='MomentumOptimizer')
    parser.add_argument(
        '--optimizer_args',
        type=json.loads,
        default='''
    {
        "learning_rate": 1e-2,
        "momentum": 0.9
    }''')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)

    # Hardware
    parser.add_argument('--train_device', default='/gpu:0')
    parser.add_argument('--eval_device', default='/gpu:0')

    # Optional comment
    parser.add_argument('--comment', default='')

    args = parser.parse_args()

    # Build name
    name = build_name(args)

    # Instantiate the model object
    model = getattr(
        importlib.import_module('models.' + args.model), args.model)()

    # Instantiate the input object
    dataset = getattr(
        importlib.import_module('inputs.' + args.dataset), args.dataset)()

    # Instantiate the optimizer
    optimizer = getattr(tf.train, args.optimizer)(**args.optimizer_args)
    print('Model name {}\nArgs: {}'.format(
        name, pprint.pformat(
            vars(args), indent=4)))

    return args, name, model, dataset, optimizer
