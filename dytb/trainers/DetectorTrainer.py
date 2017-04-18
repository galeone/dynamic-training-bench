#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Trainer for the Detector model"""

from .interfaces import Trainer


class DetectorTrainer(Trainer):
    """Trainer for the Detector model"""

    def train(self, dataset, args, steps, paths):
        """Train the model, using the dataset, utilizing the passed args
        Args:
            dataset: implementation of the Input interface
            args: dictionary of hyperparameters a train parameters
        Returns:
            info: dict containing the information of the trained model
        Side effect:
            saves the latest checkpoints and the best model in its own folder
        """
        raise ValueError("method not implemented")

    @property
    def model(self):
        """Returns the model to evaluate"""
        raise ValueError("method not implemented")

    @model.setter
    def model(self, model):
        """Set the model to evaluate.
        Args:
            model: implementation of the Model interface
        """
        raise ValueError("method not implemented")
