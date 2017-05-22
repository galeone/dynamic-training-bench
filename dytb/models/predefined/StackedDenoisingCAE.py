#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Build a stacked denoising CAE"""

import tensorflow as tf
from ..collections import LOSSES
from ..layers import conv
from ..interfaces import Autoencoder


class StackedDenoisingCAE(Autoencoder):
    """Build a stacked denoising CAE"""

    def _pad(self, input_x, filter_side):
        """
        pads input_x with the right amount of zeros.
        Args:
            input_x: 4-D tensor, [batch_side, widht, height, depth]
            filter_side: used to dynamically determine the padding amount
        Returns:
            input_x padded
        """
        # calculate the padding amount for each side
        amount = filter_side - 1
        # pad the input on top, bottom, left, right, with amount zeros
        return tf.pad(input_x, [[0, 0], [amount, amount], [amount, amount],
                                [0, 0]])

    def get(self, images, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.
        Args:
            images: model input
            num_classes: number of classes to predict. If the model doesn't use it,
                         just pass any value.
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty
        Returns:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            predictions: the model output
        """

        # Initializer with seed
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode='FAN_IN',
            uniform=False,
            seed=self.seed,
            dtype=tf.float32)

        num_layers = 9
        filter_side = 3
        filters_number = 9
        with tf.variable_scope(self.__class__.__name__):
            input_x = tf.identity(images)
            if train_phase:
                input_x_noise = tf.clip_by_value(input_x + tf.random_uniform(
                    input_x.get_shape(),
                    minval=-0.5,
                    maxval=0.5,
                    dtype=input_x.dtype,
                    seed=None), -1.0, 1.0)
            else:
                input_x_noise = input_x
            input_padded_noise = self._pad(input_x_noise, filter_side)

            for layer in range(num_layers):
                with tf.variable_scope("layer_" + str(layer)):
                    with tf.variable_scope("encode"):
                        encoding = conv(
                            input_padded_noise, [
                                filter_side, filter_side,
                                input_padded_noise.get_shape()[3].value,
                                filters_number
                            ],
                            1,
                            'VALID',
                            train_phase,
                            activation=tf.nn.relu,
                            wd=l2_penalty,
                            initializer=initializer)

                        if train_phase:
                            encoding = tf.nn.dropout(encoding, 0.5)

                    with tf.variable_scope("decode"):
                        output_x_noise = conv(
                            encoding, [
                                filter_side, filter_side, filters_number,
                                images.get_shape()[3].value
                            ],
                            1,
                            'VALID',
                            train_phase,
                            activation=tf.nn.tanh,
                            initializer=initializer)

                        last = layer == num_layers - 1
                        if train_phase and not last:
                            output_x_noise = tf.nn.dropout(output_x_noise, 0.5)

                        # loss between input without noise and output computed
                        # on noisy values
                        tf.add_to_collection(LOSSES,
                                             self._mse(output_x_noise, input_x))
                        input_x_noise = tf.stop_gradient(output_x_noise)
                        input_padded_noise = self._pad(input_x_noise,
                                                       filter_side)

        # The is_training_ placeholder is not used, but we define and return it
        # in order to respect the expected output cardinality of the get method
        is_training_ = tf.placeholder_with_default(
            False, shape=(), name="is_training_")
        return is_training_, output_x_noise

    def _mse(self, input_x, output_x):
        # 1/2n \sum^{n}_{i=i}{(x_i - x'_i)^2}
        return tf.divide(
            tf.reduce_mean(tf.square(tf.subtract(input_x, output_x))),
            2.,
            name="mse")

    def loss(self, predictions, real_values):
        """Return the loss operation between predictions and real_values.
        Add L2 weight decay term if any.
        Args:
            predictions: predicted values
            real_values: real values
        Returns:
            Loss tensor of type float.
        """
        with tf.variable_scope('loss'):
            #tf.add_to_collection(LOSSES, self._mse(real_values, predictions))
            # mse + weight_decay per layer
            error = tf.add_n(tf.get_collection(LOSSES), name='total_loss')

        return error
