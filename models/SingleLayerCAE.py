#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Build a single layer CAE"""

import tensorflow as tf
from . import utils
from .interfaces.Autoencoder import Autoencoder


class SingleLayerCAE(Autoencoder):
    """ Build a single layer CAE"""

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
        return tf.pad(input_x,
                      [[0, 0], [amount, amount], [amount, amount], [0, 0]])

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
        filter_side = 3
        filters_number = 32
        with tf.variable_scope(self.__class__.__name__):
            input_x = self._pad(images, filter_side)

            with tf.variable_scope("encode"):
                # the encoding convolutions is a [3 x 3 x input_depth] x 32 convolution
                # the activation function chosen is the tanh
                # 32 is the number of feature extracted. It's completely arbitrary as is
                # the side of the convolutional filter and the activation function used
                encoding = utils.conv_layer(
                    input_x, [
                        filter_side, filter_side, input_x.get_shape()[3].value,
                        filters_number
                    ],
                    1,
                    'VALID',
                    activation=tf.nn.tanh,
                    wd=l2_penalty)

            with tf.variable_scope("decode"):
                # the decoding convolution is a [3 x 3 x 32] x input_depth convolution
                # the activation function chosen is the tanh
                # The dimenensions of the convolutional filter in the decoding convolution,
                # differently from the encoding, are constrained by the
                # choices made in the encoding layer
                # The only degree of freedom is the chose of the activation function.
                # We have to choose an activation function that constraints the outputs
                # to live in the same space of the input values.
                # Since the input values are between -1 and 1, we can use the tanh function
                # directly, or we could use the sigmoid and then scale the output
                output_x = utils.conv_layer(
                    encoding, [
                        filter_side, filter_side, filters_number,
                        input_x.get_shape()[3].value
                    ],
                    1,
                    'VALID',
                    activation=tf.nn.tanh)

        # The is_training_ placeholder is not used, but we define and return it
        # in order to respect the expected output cardinality of the get method
        is_training_ = tf.placeholder_with_default(
            False, shape=(), name="is_training_")
        return is_training_, output_x

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
            # 1/2n \sum^{n}_{i=i}{(x_i - x'_i)^2}
            mse = tf.divide(
                tf.reduce_mean(
                    tf.square(tf.subtract(predictions, real_values))),
                2.,
                name="mse")
            tf.add_to_collection(utils.LOSSES_COLLECTION, mse)

            # mse + weight_decay per layer
            error = tf.add_n(
                tf.get_collection(utils.LOSSES_COLLECTION), name='total_loss')

        return error
