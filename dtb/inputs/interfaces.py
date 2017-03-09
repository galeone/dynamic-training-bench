#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to define an input"""

from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum, unique


class Input(object, metaclass=ABCMeta):
    """Input is the interface that classifiers must implement"""

    @abstractmethod
    def inputs(self, input_type, batch_size, augmentation_fn=None):
        """Construct input for evaluation using the Reader ops.

        Args:
            input_type: InputType enum
            batch_size: Number of elements per batch.
            augmentation_fn: function that accepts an input value,
                perform augmentation and returns the value

        Returns:
            elements:  tensor of with batch_size elements
            ground_truth: tensor with batch_size elements
        """
        pass

    @abstractmethod
    def num_examples(self, input_type):
        """Returns the number of examples for the specified input_type

        Args:
            input_type: InputType enum
        """
        pass

    @abstractproperty
    def num_classes(self):
        """Returns the number of classes"""
        pass

    @abstractproperty
    def name(self):
        """Returns the name of the input source"""
        pass


@unique
class InputType(Enum):
    """Enum to specify the data type requested"""
    validation = 'validation'
    train = 'train'
    test = 'test'

    def __str__(self):
        """Return the string representation of the enum"""
        return self.value

    @staticmethod
    def check(input_type):
        """Check if input_type is an element of this Enum"""
        if not isinstance(input_type, InputType):
            raise ValueError("Invalid input_type, required a valid InputType")
