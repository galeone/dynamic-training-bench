#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the model interfaces"""

from abc import ABCMeta, abstractmethod
# Evaluators
from ..evaluators.AutoencoderEvaluator import AutoencoderEvaluator
from ..evaluators.ClassifierEvaluator import ClassifierEvaluator
from ..evaluators.DetectorEvaluator import DetectorEvaluator
from ..evaluators.RegressorEvaluator import RegressorEvaluator
# Trainers
from ..trainers.AutoencoderTrainer import AutoencoderTrainer
from ..trainers.ClassifierTrainer import ClassifierTrainer
from ..trainers.DetectorTrainer import DetectorTrainer
from ..trainers.RegressorTrainer import RegressorTrainer


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

    @property
    def name(self):
        """Returns the name of the model"""
        return self.__class__.__name__

    @property
    def info(self):
        """Returns the inforation about the trained model"""
        return self._info

    @info.setter
    def info(self, info):
        """Save the training info
        Args:
            info: dict of training info
        """
        self._info = info

    @property
    def trainer(self):
        """Returns the trainer associated to the model"""
        obj = AutoencoderTrainer()
        obj.model = self
        return obj

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        obj = AutoencoderEvaluator()
        obj.model = self
        return obj


class Classifier(object, metaclass=ABCMeta):
    """Classifier is the interface that classifiers must implement"""

    @abstractmethod
    def get(self, images, num_classes, train_phase=False, l2_penalty=0.0):
        """Define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input
            num_classes: number of classes to predict
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Returns:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            logits: the model output
        """

    @abstractmethod
    def loss(self, logits, labels):
        """Return the loss operation between logits and labels
        Args:
            logits: Logits from get().
            labels: Labels from train_inputs or inputs(). 1-D tensor
                  of shape [batch_size]

        Returns:
            Loss tensor of type float.
        """

    @property
    def name(self):
        """Returns the name of the model"""
        return self.__class__.__name__

    @property
    def info(self):
        """Returns the inforation about the trained model"""
        return self._info

    @info.setter
    def info(self, info):
        """Save the training info
        Args:
            info: dict of training info
        """
        self._info = info

    @property
    def trainer(self):
        """Returns the trainer associated to the model"""
        obj = ClassifierTrainer()
        obj.model = self
        return obj

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        obj = ClassifierEvaluator()
        obj.model = self
        return obj


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

    @property
    def name(self):
        """Returns the name of the model"""
        return self.__class__.__name__

    @property
    def info(self):
        """Returns the inforation about the trained model"""
        return self._info

    @info.setter
    def info(self, info):
        """Save the training info
        Args:
            info: dict of training info
        """
        self._info = info

    @property
    def trainer(self):
        """Returns the trainer associated to the model"""
        obj = DetectorTrainer()
        obj.model = self
        return obj

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        obj = DetectorEvaluator()
        obj.model = self
        return obj


class Regressor(object, metaclass=ABCMeta):
    """Regressor is the interface that regressors must implement"""

    @abstractmethod
    def get(self, images, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input
            num_classes: number of classes to predict
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Returns:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            predictions: the model output
        """

    @abstractmethod
    def loss(self, predictions, labels):
        """Return the loss operation between predictions and labels
        Args:
            predictions: Predictions from get().
            labels: Labels from train_inputs or inputs(). 1-D tensor
                  of shape [batch_size]

        Returns:
            Loss tensor of type float.
        """

    @property
    def name(self):
        """Returns the name of the model"""
        return self.__class__.__name__

    @property
    def info(self):
        """Returns the inforation about the trained model"""
        return self._info

    @info.setter
    def info(self, info):
        """Save the training info
        Args:
            info: dict of training info
        """
        self._info = info

    @property
    def trainer(self):
        """Returns the trainer associated to the model"""
        obj = RegressorTrainer()
        obj.model = self
        return obj

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        obj = RegressorEvaluator()
        obj.model = self
        return obj
