#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate Classification models """

import math
import numpy as np
import tensorflow as tf
from .interfaces import Evaluator
from .metrics import accuracy_op
from ..inputs.interfaces import InputType
from ..models.utils import variables_to_restore


class ClassifierEvaluator(Evaluator):
    """ClassifierEvaluator is the evaluation object for a Classifier model"""

    def __init__(self):
        """Initialize the evaluator"""
        self._model = None

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

    def eval(self,
             checkpoint_path,
             dataset,
             input_type,
             batch_size,
             augmentation_fn=None):
        """Eval the model, restoring weight found in checkpoint_path, using the dataset.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            dataset: implementation of the Input interface
            input_type: InputType enum
            batch_size: evaluate in batch of size batch_size
            augmentation_fn: if present, applies the augmentation to the input data
        Returns:
            value: scalar value representing the evaluation of the model,
                   on the dataset, fetching values of the specified input_type
        """
        return self._accuracy(checkpoint_path, dataset, input_type, batch_size,
                              augmentation_fn)

    def stats(self, checkpoint_path, dataset, batch_size, augmentation_fn=None):
        """Run the eval method on the model, see eval for arguments
        and return value description.
        Moreover, adds informations about the model and returns the whole information
        in a dictionary.
        Returns:
            dict
        """

        train_accuracy = self.eval(checkpoint_path, dataset, InputType.train,
                                   batch_size, augmentation_fn)
        train_cm = self._confusion_matrix(checkpoint_path, dataset,
                                          InputType.train, batch_size,
                                          augmentation_fn)
        validation_accuracy = self.eval(checkpoint_path, dataset,
                                        InputType.validation, batch_size,
                                        augmentation_fn)
        validation_cm = self._confusion_matrix(checkpoint_path, dataset,
                                               InputType.validation, batch_size,
                                               augmentation_fn)
        test_accuracy = self.eval(checkpoint_path, dataset, InputType.test,
                                  batch_size, augmentation_fn)
        test_cm = validation_cm = self._confusion_matrix(
            checkpoint_path, dataset, InputType.test, batch_size,
            augmentation_fn)

        return {
            "train": {
                "accuracy": train_accuracy,
                "confusion_matrix": train_cm
            },
            "validation": {
                "accuracy": validation_accuracy,
                "confusion_matrix": validation_cm
            },
            "test": {
                "accuracy": test_accuracy,
                "confusion_matrix": test_cm
            }
        }

    def _accuracy(self,
                  checkpoint_path,
                  dataset,
                  input_type,
                  batch_size=200,
                  augmentation_fn=None):
        InputType.check(input_type)

        with tf.Graph().as_default():
            tf.set_random_seed(69)
            # Get images and labels from the dataset
            with tf.device('/cpu:0'):
                images, labels = dataset.inputs(
                    input_type=input_type,
                    batch_size=batch_size,
                    augmentation_fn=augmentation_fn)

            # Build a Graph that computes the predictions from the inference model.
            _, predictions = self._model.get(
                images, dataset.num_classes, train_phase=False)

            # Accuracy op
            accuracy = accuracy_op(predictions, labels)

            saver = tf.train.Saver(variables_to_restore())
            accuracy_value = 0.0
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    return accuracy_value

                # Start the queue runners.
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
                            dataset.num_examples(input_type) / batch_size))
                    # Counts the number of correct predictions.
                    accuracy_sum = 0.0
                    step = 0
                    while step < num_iter and not coord.should_stop():
                        accuracy_sum += sess.run(accuracy)
                        step += 1

                    accuracy_value = accuracy_sum / step
                except Exception as exc:
                    coord.request_stop(exc)
                finally:
                    coord.request_stop()

                coord.join(threads)
            return accuracy_value

    def _confusion_matrix(self,
                          checkpoint_path,
                          dataset,
                          input_type,
                          batch_size=200,
                          augmentation_fn=None):
        InputType.check(input_type)

        with tf.Graph().as_default():
            # Get images and labels from the dataset
            with tf.device('/cpu:0'):
                images, labels = dataset.inputs(
                    input_type=input_type,
                    batch_size=batch_size,
                    augmentation_fn=augmentation_fn)

            # Build a Graph that computes the predictions from the inference model.
            _, predictions = self._model.get(
                images, dataset.num_classes, train_phase=False)

            # handle fully convolutional classifiers
            predictions_shape = predictions.shape
            if len(predictions_shape) == 4 and predictions_shape[1:3] == [1, 1]:
                top_k_predictions = tf.squeeze(predictions, [1, 2])
            else:
                top_k_predictions = predictions

            # Extract the predicted label (top-1)
            _, top_predicted_label = tf.nn.top_k(
                top_k_predictions, k=1, sorted=False)
            # (batch_size, k) -> k = 1 -> (batch_size)
            top_predicted_label = tf.squeeze(top_predicted_label, axis=1)

            confusion_matrix_op = tf.confusion_matrix(
                labels, top_predicted_label, num_classes=dataset.num_classes)

            saver = tf.train.Saver(variables_to_restore())
            confusion_matrix = np.zeros(
                (dataset.num_classes, dataset.num_classes), dtype=np.int64)
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    return confusion_matrix

                # Start the queue runners.
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
                            dataset.num_examples(input_type) / batch_size))

                    # Accumulate the confusion matrices for batch
                    step = 0
                    while step < num_iter and not coord.should_stop():
                        confusion_matrix += sess.run(confusion_matrix_op)
                        step += 1

                except Exception as exc:
                    coord.request_stop(exc)
                finally:
                    coord.request_stop()

                coord.join(threads)
            return confusion_matrix

    def extract_features(self, checkpoint_path, inputs, layer_name,
                         num_classes):
        """Restore model parameters from checkpoint_path. Search in the model
        the layer with name `layer_name`. If found places `inputs` as input to the model
        and returns the values extracted by the layer.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            inputs: a Tensor with a shape compatible with the model's input
            layer_name: a string, the name of the layer to extract from model
            num_classes: number of classes to classify, this number must be equal to the number
            of classes the classifier was trained on, otherwise the restore from checkpoint fails
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
            tf.set_random_seed(69)

            inputs_ = tf.placeholder(inputs.dtype, shape=inputs.shape)

            # Build a Graph that computes the predictions from the inference model.
            _ = self._model.get(inputs_, num_classes, train_phase=False)

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
