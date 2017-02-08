#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to work with Autoencoders"""

from abc import ABCMeta, abstractmethod


class Autoencoder(object, metaclass=ABCMeta):
    """Autoencoder is the interface that classifiers must implement"""

    @abstractmethod
    def get(self, images, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Returns:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            predictions: the model output
        """

    @abstractmethod
    def loss(self, predictions, real_values):
        """Return the loss operation between predictions and real_values
        Args:
          predictions: predicted values
          labels: real_values

        Returns:
          Loss tensor of type float.
        """
