#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to define a trainer"""

from abc import ABCMeta, abstractmethod


class Trainer(object, metaclass=ABCMeta):
    """Trainer is the interface that a trainer must implement"""

    @abstractmethod
    def train(self, dataset, args):
        """Train the model, using the dataset, utilizing the passed args
        Args:
            dataset: implementation of the Input interface
            args: dictionary of hyperparameters a train parameters

        Returns:
            info: dict containing the information of the trained model
        """

    @abstractmethod
    def set_model(self, model):
        """Set the model to evaluate.
        Args:
            model: implementation of the Model interface
        """
