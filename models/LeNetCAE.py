#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Build a single layer CAE with the first conv layer of LeNet """

import tensorflow as tf
from . import utils
from .Autoencoder import Autoencoder


class LeNetCAE(Autoencoder):
    """ Build a single layer CAE with the first conv layer of LeNet """

    def _inference(self,
                   images,
                   is_training_,
                   train_phase=False,
                   l2_penalty=0.0):
        """Build the LeNet-like network.

        Args:
          images: Images returned from distorted_inputs() or inputs().
          is_training_: enable/disable training ops at run time
          train_phase: Boolean to enable/disable training ops at build time
          l2_penalty: float value, weight decay (l2) penalty

        Returns:
          [batch_size, height, widht, depth] reconstrued images
        """

        def pad(input_x, filter_side):
            """
            pads input_x with the right amount of zeros.
            Args:
                input_x: 4-D tensor, [batch_side, widht, height, depth]
             filter_side: used to dynamically determine the padding amount
            Returns:
                input_x padded
            """
            import math
            amount = math.ceil(filter_side / 2) + 1
            return tf.pad(input_x,
                          [[0, 0], [amount, amount], [amount, amount], [0, 0]])

        with tf.variable_scope(self.__class__.__name__):
            input_x = pad(images, 5)

            with tf.variable_scope("encode"):
                encoding = tf.nn.tanh(
                    utils.conv_layer(
                        input_x, [5, 5, input_x.get_shape()[3].value, 32],
                        1,
                        'VALID',
                        wd=l2_penalty))

            with tf.variable_scope("decode"):
                output_x = tf.nn.tanh(
                    utils.conv_layer(
                        encoding, [5, 5, 32, input_x.get_shape()[3].value],
                        1,
                        'VALID',
                        wd=l2_penalty))

            return output_x

    def loss(self, predictions, real_values):
        """Add L2Loss to all the trainable variables.
        Args:
          predictions: predicted values
          labels: real_values

        Returns:
          Loss tensor of type float.
        """
        with tf.variable_scope('loss'):
            mse = tf.reduce_mean(
                tf.nn.l2_loss(
                    tf.cast(predictions, tf.float32) - tf.cast(real_values,
                                                               tf.float32)),
                name="mse")
            tf.add_to_collection('losses', mse)

            # The total loss is defined as the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            error = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return error

    def get(self, images, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Return:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            predictions: the model output
        """
        is_training_ = tf.placeholder(tf.bool, shape=(), name="is_training_")
        predictions = self._inference(images, is_training_, train_phase,
                                      l2_penalty)
        return is_training_, predictions
