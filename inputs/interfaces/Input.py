#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to defined an input"""

from abc import ABCMeta, abstractmethod, abstractproperty


class Input(object, metaclass=ABCMeta):
    """Input is the interface that classifiers must implement"""

    @abstractmethod
    def distorted_inputs(self, batch_size):
        """Construct distorted input for training using the Reader ops.

        Args:
            batch_size: Number of elements per batch.

        Returns:
            elements: distorted elements. Tensor of with batch_size elements
            ground_truth: tensor with batch_size elements
        """
        pass

    @abstractmethod
    def inputs(self, input_type, batch_size):
        """Construct input for evaluation using the Reader ops.

        Args:
            input_type: InputType enum
            batch_size: Number of elements per batch.

        Returns:
            elements:  tensor of with batch_size elements
            ground_truth: tensor with batch_size elements
        """
        pass

    @abstractmethod
    def num_examples(self, input_type):
        """Returns the number of examples per the specified input_type

        Args:
            input_type: InputType enum
        """
        pass

    @abstractproperty
    def num_classes(self):
        """Returns the number of classes """
        pass
