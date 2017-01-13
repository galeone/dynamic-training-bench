#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to defined an input"""

import abc


class Input(object, metaclass=abc.ABCMeta):
    """Input is the interface that classifiers must implement"""

    @abc.abstractmethod
    def distorted_inputs(self, batch_size):
        """Construct distorted input for training using the Reader ops.

        Args:
            batch_size: Number of images per batch.

        Returns:
            images: distorted images. 4D tensor of [batch_size, self._image_height, self._image_width, self._image_depth] size.
            ground_truth: tensor with batch_size elements
        """
        raise NotImplementedError(
            'users must define distorted_inputs to use this base class')

    @abc.abstractmethod
    def inputs(self, input_type, batch_size):
        """Construct input for evaluation using the Reader ops.

        Args:
            input_type: InputType enum
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, self._image_height, self._image_width, self._image_depth] size.
            ground_truth: tensor with batch_size elements
        """
        raise NotImplementedError(
            'users must define inputs to use this base class')

    @abc.abstractmethod
    def maybe_download_and_extract(self):
        """Acquire and save the dataset"""
        raise NotImplementedError(
            'users must define maybe_download_and_extract to use this base class'
        )

    @abc.abstractmethod
    def num_examples(self, input_type):
        """Returns the number of examples per the specified input_type

        Args:
            input_type: InputType enum
        """
        raise NotImplementedError(
            'users must define num_examples to use this base class')

    @abc.abstractmethod
    def num_classes(self):
        """Returns the number of classes """
        raise NotImplementedError(
            'users must define num_classes to use this base class')
