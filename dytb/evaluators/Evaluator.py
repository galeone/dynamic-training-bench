#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Define the interface to implement to define an evaluator"""

import math
from abc import abstractproperty, ABCMeta
import numpy as np
import tensorflow as tf
from ..inputs.interfaces import InputType
from ..models.utils import variables_to_restore


class Evaluator(object, metaclass=ABCMeta):
    """Evaluator is the class in charge of evaluate the models"""

    def __init__(self):
        self._model = None
        self._dataset = None
        self._visualizations = []

    @property
    def model(self):
        """Returns the model to evaluate"""
        return self._model

    @model.setter
    def model(self, model):
        """Set the model to evaluate.
        Args:
            model: implementation of the Model interface
        """
        self._model = model

    @property
    def dataset(self):
        """Returns the dataset to use to evaluate the model"""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """Set the dataset to use to evaluate the model
        Args:
            dataset: implementation of the Input interface
        """
        self._dataset = dataset

    @property
    def visualizations(self):
        """Returns a list of dict with keys:
        {
            "fn": function(inputs, predictions, targets) that returns an image
            "name": name
        }
        """
        return self._visualizations

    @visualizations.setter
    def visualizations(self, visualizations):
        """Set the visualization list to disply
        Args:
            visualizations: the list of visualizations
        """
        self._visualizations = visualizations

    @abstractproperty
    def metrics(self):
        """Returns a list of dict with keys:
        {
            "fn": function(predictions, targets)
            "name": name
            "positive_trend_sign": sign that we like to see when things go well
            "model_selection": boolean, True if the metric has to be measured to select the model
            "average": boolean, true if the metric should be computed as average over the batches.
                       If false the results over the batches are just added
            "tensorboard": boolean. True if the metric is a scalar and can be logged in tensoboard
        }
        """

    def eval(self,
             metric,
             checkpoint_path,
             input_type,
             batch_size,
             augmentation_fn=None):
        """Eval the model, restoring weight found in checkpoint_path, using the dataset.
        Args:
            metric: the metric to evaluate, a single element of self.metrics
            checkpoint_path: path of the trained model checkpoint directory
            input_type: InputType enum
            batch_size: evaluate in batch of size batch_size
            augmentation_fn: if present, applies the augmentation to the input data

        Returns:
            value: scalar value representing the evaluation of the metric on the restored model
                   on the dataset, fetching values of the specified input_type.
        """
        InputType.check(input_type)

        with tf.Graph().as_default():
            # Get inputs and targets: inputs is an input batch
            # target could be either an array of elements or a tensor.
            # it could be [label] or [label, attr1, attr2, ...]
            # or Tensor, where tensor is a standard tensorflow Tensor with
            # its own shape
            with tf.device('/cpu:0'):
                inputs, *targets = self.dataset.inputs(
                    input_type=input_type,
                    batch_size=batch_size,
                    augmentation_fn=augmentation_fn)

            # Build a Graph that computes the predictions from the
            # inference model.
            # Preditions is an array of predictions with the same cardinality of
            # targets
            _, *predictions = self._model.get(
                inputs,
                self.dataset.num_classes,
                train_phase=False,
                l2_penalty=0.0)

            if len(predictions) != len(targets):
                print(("{}.get 2nd return value and {}.inputs 2nd return "
                       "value must have the same cardinality but got: {} vs {}"
                      ).format(self._model.name, self.dataset.name,
                               len(predictions), len(targets)))
                return

            if len(predictions) == 1:
                predictions = predictions[0]
                targets = targets[0]

            metric_fn = metric["fn"](predictions, targets)

            saver = tf.train.Saver(variables_to_restore())
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    sign = math.copysign(1, metric["positive_trend_sign"])
                    return float('inf') if sign < 0 else float("-inf")

                # Start the queue runners
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for queue_runner in tf.get_collection(
                            tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(
                            queue_runner.create_threads(
                                sess, coord=coord, daemon=True, start=True))

                    num_iter = int(
                        math.ceil(
                            self.dataset.num_examples(input_type) / batch_size))
                    step = 0
                    metric_value_sum = 0.0
                    while step < num_iter and not coord.should_stop():
                        step += 1
                        value = sess.run(metric_fn)
                        # metrics can sometimes have NaN
                        # (think about a metric that excludes a certain class and the input batch
                        # has only element of that class into)
                        # NaN, not being a number, are excluded from the calculation of the metric
                        if math.isnan(value):
                            step -= 1
                        else:
                            metric_value_sum += value
                    avg_metric_value = metric_value_sum / step if metric[
                        "average"] else metric_value_sum
                except Exception as exc:
                    coord.request_stop(exc)
                finally:
                    coord.request_stop()

                coord.join(threads)
            return avg_metric_value

    def stats(self, checkpoint_path, batch_size, augmentation_fn=None):
        """Run the eval method on the model, see eval for arguments
        and return value description.
        Moreover, adds informations about the model and returns the whole information
        in a dictionary.
        Returns:
            dict
        """
        return {
            "train": {
                metric["name"]:
                self.eval(metric, checkpoint_path, InputType.train, batch_size,
                          augmentation_fn)
                for metric in self.metrics
            },
            "validation": {
                metric["name"]:
                self.eval(metric, checkpoint_path, InputType.validation,
                          batch_size, augmentation_fn)
                for metric in self.metrics
            },
            "test": {
                metric["name"]:
                self.eval(metric, checkpoint_path, InputType.test, batch_size,
                          augmentation_fn)
                for metric in self.metrics
            },
        }

    def visualize(self,
                  viz,
                  checkpoint_path,
                  input_type,
                  batch_size,
                  augmentation_fn=None):
        """Restore the model, restoring weight found in checkpoint_path, using the dataset.
        Execute the function **for a single step**.
        Args:
            viz: the function to evaluate, a single element of self.visualizations
            checkpoint_path: path of the trained model checkpoint directory
            input_type: InputType enum
            batch_size: evaluate in batch of size batch_size
            augmentation_fn: if present, applies the augmentation to the input data

        Returns:
            image: a numpy batch of images
        """
        InputType.check(input_type)

        with tf.Graph().as_default():
            # Get inputs and targets: inputs is an input batch
            # target could be either an array of elements or a tensor.
            # it could be [label] or [label, attr1, attr2, ...]
            # or Tensor, where tensor is a standard tensorflow Tensor with
            # its own shape
            with tf.device('/cpu:0'):
                inputs, *targets = self.dataset.inputs(
                    input_type=input_type,
                    batch_size=batch_size,
                    augmentation_fn=augmentation_fn)

            # Build a Graph that computes the predictions from the
            # inference model.
            # Preditions is an array of predictions with the same cardinality of
            # targets
            _, *predictions = self._model.get(
                inputs,
                self.dataset.num_classes,
                train_phase=False,
                l2_penalty=0.0)

            if len(predictions) != len(targets):
                print(("{}.get 2nd return value and {}.inputs 2nd return "
                       "value must have the same cardinality but got: {} vs {}"
                      ).format(self._model.name, self.dataset.name,
                               len(predictions), len(targets)))
                return

            if len(predictions) == 1:
                predictions = predictions[0]
                targets = targets[0]

            saver = tf.train.Saver(variables_to_restore())

            viz_fn = viz["fn"](inputs, predictions, targets)
            init = [
                tf.variables_initializer(tf.global_variables() +
                                         tf.local_variables()),
                tf.tables_initializer()
            ]
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    return None

                # Start the queue runners
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for queue_runner in tf.get_collection(
                            tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(
                            queue_runner.create_threads(
                                sess, coord=coord, daemon=True, start=True))

                    sess.run(init)
                    return sess.run(viz_fn)
                except Exception as exc:
                    coord.request_stop(exc)
                finally:
                    coord.request_stop()

                coord.join(threads)
        return None

    def extract_features(self,
                         checkpoint_path,
                         inputs,
                         layer_name,
                         num_classes=0):
        """Restore model parameters from checkpoint_path. Search in the model
        the layer with name `layer_name`. If found places `inputs` as input to the model
        and returns the values extracted by the layer.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            inputs: a Tensor with a shape compatible with the model's input
            layer_name: a string, the name of the layer to extract from model
            num_classes: number of classes to classify, this number must be equal to the number
            of classes the classifier was trained on, if the model is a classifier or however is
            a model class aware, otherwise let the number = 0
        Returns:
            features: a numpy ndarray that contains the extracted features
        """

        # Evaluate the inputs in the current default graph
        # then user a placeholder to inject the computed values into the new graph
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            evaluated_inputs = sess.run(inputs)

        # Create a new graph to not making dirty the default graph after subsequent
        # calls
        with tf.Graph().as_default() as graph:
            inputs_ = tf.placeholder(inputs.dtype, shape=inputs.shape)

            # Build a Graph that computes the predictions from the inference model.
            _ = self._model.get(
                inputs, num_classes, train_phase=False, l2_penalty=0.0)

            # This will raise an exception if layer_name is not found
            layer = graph.get_tensor_by_name(layer_name)

            saver = tf.train.Saver(variables_to_restore())
            features = np.zeros(layer.shape)
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    return features

                features = sess.run(
                    layer, feed_dict={inputs_: evaluated_inputs})

            return features
