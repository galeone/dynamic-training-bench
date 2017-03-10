#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate Autoencoding models """

import math
import tensorflow as tf
from . import metrics
from .interfaces import Evaluator
from ..inputs.interfaces import InputType, Input
from ..models.utils import variables_to_restore


class AutoencoderEvaluator(Evaluator):
    """AutoencoderEvaluator is the evaluation object for a Autoencoder model"""

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

    def eval(self, checkpoint_path, dataset, input_type, batch_size):
        """Eval the model, restoring weight found in checkpoint_path, using the dataset.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            dataset: implementation of the Input interface
            input_type: InputType enum
            batch_size: evaluate in batch of size batch_size

        Returns:
            value: scalar value representing the evaluation of the model,
                   on the dataset, fetching values of the specified input_type
        """
        return self._error(checkpoint_path, dataset, input_type, batch_size)

    def stats(self, checkpoint_path, dataset, batch_size):
        """Run the eval method on the model, see eval for arguments
        and return value description.
        Moreover, adds informations about the model and returns the whole information
        in a dictionary.
        Returns:
            dict
        """

        train_error = self.eval(checkpoint_path, dataset, InputType.train,
                                batch_size)
        validation_error = self.eval(checkpoint_path, dataset,
                                     InputType.validation, batch_size)
        test_error = self.eval(checkpoint_path, dataset, InputType.test,
                               batch_size)

        return {
            "train": train_error,
            "validation": validation_error,
            "test": validation_error,
            "dataset": dataset.name,
            "model": self._model.name
        }

    def _error(self, checkpoint_path, dataset, input_type, batch_size=200):
        """
        Reads the checkpoint and use it to evaluate the model
        Args:
            checkpoint_path: checkpoint folder
            dataset: python package containing the dataset to use
            input_type: InputType enum, the input type of the input examples
            batch_size: batch size for the evaluation in batches
        Returns:
            average_error: the average error
        """
        InputType.check(input_type)

        with tf.Graph().as_default():
            # Get images and labels from the dataset
            with tf.device('/cpu:0'):
                images, labels = dataset.inputs(
                    input_type=input_type, batch_size=batch_size)

            # Build a Graph that computes the predictions from the inference model.
            _, predictions = self._model.get(
                images, train_phase=False, l2_penalty=0.0)
            loss = self._model.loss(predictions, images)

            saver = tf.train.Saver(variables_to_restore())
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    return

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
                    step = 0
                    average_error = 0.0
                    while step < num_iter and not coord.should_stop():
                        error_value = sess.run(loss)
                        step += 1
                        average_error += error_value

                    average_error /= step
                except Exception as exc:
                    coord.request_stop(exc)
                finally:
                    coord.request_stop()

                coord.join(threads)
            return average_error
