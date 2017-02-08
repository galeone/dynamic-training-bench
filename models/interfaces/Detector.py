#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to work with detectors"""

from abc import ABCMeta, abstractmethod


class Detector(object, metaclass=ABCMeta):
    """Detector is the interface that detectors must implement"""

    @abstractmethod
    def get(self, images, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input, tensor with batch_size elements
            num_classes: number of classes to predict
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Returns:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            logits: the unscaled prediction for a class specific detector
            bboxes: the predicted coordinates for every detected object in the input image
                    this must have the same number of rows of logits
        """

    @abstractmethod
    def loss(self, label_relations, bboxes_relations):
        """Return the loss operation.
        Args:
            label_relations: a tuple with 2 elements, usually the pair
            (labels, logits), each one a tensor of batch_size elements
            bboxes_relations: a tuple with 2 elements, usually the pair
            (coordinates, bboxes) where coordinates are the
            ground truth coordinates ad bboxes the predicted one
        Returns:
            Loss tensor of type float.
        """
