#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Class that defines and parse CLI arguments"""

import glob
import argparse
import json
import importlib
import pprint
import tensorflow as tf


class CLIArgs(object):
    """Class that defines and parse CLI arguments"""

    def __init__(self, description="Train the model"):
        """Initialize variables:
        Args:
            description: The description to show when the help is displayed"""
        self._description = description
        self._args = None

    def _build_name(self):
        """Build method name parsing args"""
        optimizer = getattr(tf.train, self._args.optimizer)(
            **self._args.optimizer_args)
        learning_rate = self._args.optimizer_args["learning_rate"]
        name = '{}_{}_lr={}_'.format(self._args.dataset,
                                     optimizer.get_name(), learning_rate)

        if self._args.lr_decay:
            name += 'exp_lr_'
        if self._args.l2_penalty != 0.0:
            name += 'l2={}_'.format(self._args.l2_penalty)
        if self._args.comment != '':
            name += '{}_'.format(self._args.comment)

        return name.rstrip('_')

    @staticmethod
    def get_models():
        """Returns the avaiable modules filename, without the .py ext"""
        return [
            model[len('models/'):-3] for model in glob.glob('models/*.py')
            if "__init__.py" not in model and "utils" not in model
        ]

    @staticmethod
    def get_datasets():
        """Returns the avaiable datasets filename, without the .py ext"""
        return [
            dataset[len('inputs/'):-3] for dataset in glob.glob('inputs/*.py')
            if "__init__.py" not in dataset and "utils" not in dataset
        ]

    @staticmethod
    def get_optimizers():
        """Returns the avaiable Tensorflow optimizer"""
        return [
            optimizer for optimizer in dir(tf.train)
            if optimizer.endswith("Optimizer")
        ]

    def _init_parser(self):
        """Parse CLI flags shared by train & eval proceudres.
        Returns:
            parser: parser object"""

        # CLI arguments
        parser = argparse.ArgumentParser(description=self._description)

        # Required arguments
        parser.add_argument('--model', required=True, choices=self.get_models())
        parser.add_argument(
            '--dataset', required=True, choices=self.get_datasets())

        parser.add_argument('--batch_size', type=int, default=128)

        return parser

    def _get_model_dataset(self):
        """Return the model object and the dataset object.
        Returns:
            model: model object instantiated
            dataset: input object instantiated"""

        # Instantiate the model object
        model = getattr(
            importlib.import_module('models.' + self._args.model),
            self._args.model)()

        # Instantiate the input object
        dataset = getattr(
            importlib.import_module('inputs.' + self._args.dataset),
            self._args.dataset)()
        return model, dataset

    def parse_eval(self):
        """Parser the CLI arguments for the evaluation procedure
        and return
        Returns:
            args: args object
            model: model object instantiated
            dataset: input object instantiated"""

        parser = self._init_parser()
        parser.add_argument(
            "--checkpoint_path",
            required=True,
            help='the path to a checkpoint from which load the model')
        parser.add_argument("--test", action="store_true", help='use test set')

        # Hardware
        parser.add_argument('--eval_device', default='/gpu:0')
        self._args = parser.parse_args()
        # Get model and dataset objects
        model, dataset = self._get_model_dataset()
        return self._args, model, dataset

    def parse_train(self):
        """Parser the CLI arguments for the training procedure
        and return
        Returns:
            args: args object
            model name: string representing the model name
            model: model object instantiated
            dataset: input object instantiated
        """

        parser = self._init_parser()

        # Restart train or continue
        parser.add_argument(
            '--restart',
            action='store_true',
            help='restart the training process DELETING the old checkpoint files'
        )

        # Learning rate decay arguments
        parser.add_argument(
            '--lr_decay',
            action='store_true',
            help='enable the learning rate decay')
        parser.add_argument(
            '--lr_decay_epochs',
            type=int,
            default=25,
            help='decay the learning rate every lr_decay_epochs epochs')
        parser.add_argument(
            '--lr_decay_factor',
            type=float,
            default=0.1,
            help='decay of lr_decay_factor the initial learning rate after lr_decay_epochs epochs'
        )

        # L2 regularization arguments
        parser.add_argument(
            '--l2_penalty',
            type=float,
            default=0.0,
            help='L2 penalty term to apply ad the trained parameters')

        # Optimization arguments
        parser.add_argument(
            '--optimizer',
            choices=self.get_optimizers(),
            default='MomentumOptimizer',
            help='the optimizer to use')
        parser.add_argument(
            '--optimizer_args',
            type=json.loads,
            default='''
        {
            "learning_rate": 1e-2,
            "momentum": 0.9
        }''',
            help='the optimizer parameters')
        parser.add_argument(
            '--epochs',
            type=int,
            default=150,
            help='number of epochs to train the model')

        # Hardware
        parser.add_argument(
            '--train_device',
            default='/gpu:0',
            help='the device on which place the the model during the trining phase'
        )

        # Optional comment
        parser.add_argument(
            '--comment',
            default='',
            help='comment string to preprend to the model name')

        # Fine tuning & graph manipulation
        parser.add_argument(
            '--exclude_scopes',
            help='comma separated list of scopes of variables to exclude from the checkpoint restoring.',
            default=[],
            type=lambda scope_list: [scope.strip() for scope in scope_list.split(',')])

        parser.add_argument(
            '--trainable_scopes',
            help='comma separated list of scopes of variables to train. If empty every variable is trained',
            default=[],
            type=lambda scope_list: [scope.strip() for scope in scope_list.split(',')])

        parser.add_argument(
            "--checkpoint_path",
            required=False,
            default='',
            help='the path to a checkpoint from which load the model')

        # Build the object
        self._args = parser.parse_args()

        # Build name
        name = self._build_name()

        # Get model and dataset objects
        model, dataset = self._get_model_dataset()

        print('Model name {}\nArgs: {}'.format(
            name, pprint.pformat(vars(self._args), indent=4)))

        return self._args, name, model, dataset
