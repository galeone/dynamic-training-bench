#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Class that defines and parse CLI arguments"""

import os
import glob
import argparse
import json
import importlib
import pprint
import sys
import tensorflow as tf


class CLIArgs(object):
    """Class that defines and parse CLI arguments"""

    def __init__(self, description="Train the model"):
        """Initialize variables:
        Args:
            description: The description to show when the help is displayed"""
        self._description = description
        self._args = None

    @staticmethod
    def get_dytb_models():
        """Returns the avaiable dytb modules filename, without the .py ext"""
        dytbmodels_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.path.pardir,
            'models', 'predefined')
        dytbmodels = [
            model[len(dytbmodels_dir) + 1:-3]
            for model in glob.glob('{}/*.py'.format(dytbmodels_dir))
            if "__init__.py" not in model
        ]
        return dytbmodels

    @staticmethod
    def get_dytb_datasets():
        """Returns the avaiable dytb datasets filename, without the .py ext"""
        dytbdatasets_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.path.pardir,
            'inputs', 'predefined')
        dytbdatasets = [
            dataset[len(dytbdatasets_dir) + 1:-3]
            for dataset in glob.glob('{}/*.py'.format(dytbdatasets_dir))
            if "__init__.py" not in dataset
        ]
        return dytbdatasets

    @staticmethod
    def get_local_models():
        """Returns the avaiable modules filename, without the .py ext"""
        models_dir = os.path.join(os.getcwd(), 'models')
        return [
            model[len(models_dir) + 1:-3]
            for model in glob.glob('{}/*.py'.format(models_dir))
            if "__init__.py" not in model
        ]

    @staticmethod
    def get_local_datasets():
        """Returns the avaiable datasets filename, without the .py ext"""
        datasets_dir = os.path.join(os.getcwd(), 'inputs')
        return [
            dataset[len(datasets_dir) + 1:-3]
            for dataset in glob.glob('{}/*.py'.format(datasets_dir))
            if "__init__.py" not in dataset
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
        parser.add_argument(
            '--model',
            required=True,
            choices=self.get_dytb_models() + self.get_local_models())
        parser.add_argument(
            '--dataset',
            required=True,
            choices=self.get_dytb_datasets() + self.get_local_datasets())
        parser.add_argument('--batch_size', type=int, default=128)

        return parser

    def _get_model_dataset(self):
        """Return the model object and the dataset object.
        Returns:
            model: model object instantiated
            dataset: input object instantiated"""

        sys.path.append(os.getcwd())

        # Instantiate the model object
        # Give the precedence to local models
        if self._args.model in self.get_local_models():
            model = getattr(
                importlib.import_module('models.' + self._args.model),
                self._args.model)()
        else:
            model = getattr(
                importlib.import_module('dytb.models.predefined.' +
                                        self._args.model), self._args.model)()

        # Instantiate the input object
        # Give the precedente to local datasets
        if self._args.dataset in self.get_local_datasets():
            dataset = getattr(
                importlib.import_module('inputs.' + self._args.dataset),
                self._args.dataset)()
        else:
            dataset = getattr(
                importlib.import_module('dytb.inputs.predefined.' +
                                        self._args.dataset),
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
            help=
            'decay of lr_decay_factor the initial learning rate after lr_decay_epochs epochs'
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
            help=
            'the device on which place the the model during the trining phase')

        # Optional comment
        parser.add_argument(
            '--comment',
            default='',
            help='comment string to preprend to the model name')

        # Fine tuning & graph manipulation
        parser.add_argument(
            '--exclude_scopes',
            help='comma separated list of scopes of variables to exclude from the checkpoint restoring.',
            default=None,
            type=lambda scope_list: [scope.strip() for scope in scope_list.split(',')])

        parser.add_argument(
            '--trainable_scopes',
            help='comma separated list of scopes of variables to train. If empty every variable is trained',
            default=None,
            type=lambda scope_list: [scope.strip() for scope in scope_list.split(',')])

        parser.add_argument(
            "--checkpoint_path",
            required=False,
            default='',
            help='the path to a checkpoint from which load the model')

        # Build the object
        self._args = parser.parse_args()

        # Get model and dataset objects
        model, dataset = self._get_model_dataset()

        print('Args: {}'.format(pprint.pformat(vars(self._args), indent=4)))

        return self._args, model, dataset
