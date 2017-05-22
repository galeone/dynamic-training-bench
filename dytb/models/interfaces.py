#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the model interfaces"""

from abc import ABCMeta, abstractmethod, abstractproperty
# Evaluators
from ..evaluators.AutoencoderEvaluator import AutoencoderEvaluator
from ..evaluators.ClassifierEvaluator import ClassifierEvaluator
from ..evaluators.DetectorEvaluator import DetectorEvaluator
from ..evaluators.RegressorEvaluator import RegressorEvaluator


class Autoencoder(object, metaclass=ABCMeta):
    """Autoencoder is the interface that classifiers must implement"""

    def __init__(self):
        self._info = {}
        self._seed = None
        self._evaluator = None

    @abstractmethod
    def get(self, inputs, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            inputs: model input
            num_classes: number of classes to predict. If the model doesn't use it,
                         just pass any value.
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
    def seed(self):
        """Returns the seed used for weight initialization"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed to use for weight initialization
        Args:
            seed
        """
        self._seed = seed

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        if self._evaluator is None:
            obj = AutoencoderEvaluator()
            obj.model = self
            self._evaluator = obj

        return self._evaluator


class Classifier(object, metaclass=ABCMeta):
    """Classifier is the interface that classifiers must implement"""

    def __init__(self):
        self._info = {}
        self._seed = None
        self._evaluator = None

    @abstractmethod
    def get(self, inputs, num_classes, train_phase=False, l2_penalty=0.0):
        """Define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            inputs: model input
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
    def seed(self):
        """Returns the seed used for weight initialization"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed to use for weight initialization
        Args:
            seed
        """
        self._seed = seed

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        if self._evaluator is None:
            obj = ClassifierEvaluator()
            obj.model = self
            self._evaluator = obj
        return self._evaluator


class Detector(object, metaclass=ABCMeta):
    """Detector is the interface that detectors must implement"""

    def __init__(self):
        self._info = {}
        self._seed = None
        self._evaluator = None

    @abstractmethod
    def get(self, inputs, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            inputs: model input, tensor with batch_size elements
            num_classes: number of classes to predict. If the model doesn't use it,
                         just pass any value.
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
    def seed(self):
        """Returns the seed used for weight initialization"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed to use for weight initialization
        Args:
            seed
        """
        self._seed = seed

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        if self._evaluator is None:
            obj = DetectorEvaluator()
            obj.model = self
            self._evaluator = obj
        return self._evaluator


class Regressor(object, metaclass=ABCMeta):
    """Regressor is the interface that regressors must implement"""

    def __init__(self):
        self._info = {}
        self._seed = None
        self._evaluator = None

    @abstractmethod
    def get(self, inputs, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            inputs: model input
            num_classes: number of classes to predict. If the model doesn't use it,
                         just pass any value.
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
    def seed(self):
        """Returns the seed used for weight initialization"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed to use for weight initialization
        Args:
            seed
        """
        self._seed = seed

    @property
    def evaluator(self):
        """Returns the evaluator associated to the model"""
        if self._evaluator is None:
            obj = RegressorEvaluator()
            obj.model = self
            self._evaluator = obj
        return self._evaluator


class Custom(object, metaclass=ABCMeta):
    """Custom is the interface that custom models must implement"""

    def __init__(self):
        self._info = {}
        self._seed = None
        self._evaluator = None

    @abstractmethod
    def get(self, inputs, num_classes, **kwargs):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            inputs: model input
            num_classes: number of classes to predict. If the model doesn't use it,
                         just pass any value.
            kwargs:
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
            predictions: a list of predicted values eg [predicted_labels_batch, ...]
            labels: a list of real_values, eg [ labels_batch, attributeA_batch, ...]

        Returns:
            Loss tensor of type float.
        """

    @abstractproperty
    def evaluator(self):
        """Returns the evaluator associated to the model"""

    # Below implemented properties

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
    def seed(self):
        """Returns the seed used for weight initialization"""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed to use for weight initialization
        Args:
            seed
        """
        self._seed = seed
