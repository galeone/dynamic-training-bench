#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate Detection models """

import numpy as np
import tensorflow as tf
from .interfaces import Evaluator
from ..models.utils import variables_to_restore


class DetectorEvaluator(Evaluator):
    """DetectorEvaluator is the evaluation object for a Detector model"""

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
        raise ValueError("method not implemented")

    def stats(self, checkpoint_path, dataset, batch_size, augmentation_fn=None):
        """Run the eval method on the model, see eval for arguments
        and return value description.
        Moreover, adds informations about the model and returns the whole information
        in a dictionary.
        Returns:
            dict
        """
        raise ValueError("method not implemented")

    def extract_features(self, checkpoint_path, inputs, layer_name):
        """Restore model parameters from checkpoint_path. Search in the model
        the layer with name `layer_name`. If found places `inputs` as input to the model
        and returns the values extracted by the layer.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            inputs: a Tensor with a shape compatible with the model's input
            layer_name: a string, the name of the layer to extract from model
        Returns:
            features: a numpy ndarray that contains the extracted features
        """

        # Evaluate the inputs in the current default graph
        # then user a placeholder to inject the computed values into the new graph
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            evaluated_inputs = sess.run(inputs)

        with tf.Graph().as_default() as graph:
            tf.set_random_seed(69)

            inputs_ = tf.placeholder(inputs.dtype, shape=inputs.shape)

            # Build a Graph that computes the predictions from the inference model.
            _ = self._model.get(inputs, train_phase=False)

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
