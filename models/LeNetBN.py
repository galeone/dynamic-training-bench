#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Build a LeNet-like network with BN layers"""

import tensorflow as tf
from . import utils
from .interfaces.Classifier import Classifier


class LeNetBN(Classifier):
    """Build a LeNet-like network with BN layers"""

    def _inference(self,
                   images,
                   num_classes,
                   is_training_,
                   train_phase=False,
                   l2_penalty=0.0):
        """Build the LeNet-like network.

        Args:
          images: Images returned from distorted_inputs() or inputs().
          num_classes: Number of classes to predict
          is_training_: enable/disable training ops at run time
          train_phase: Boolean to enable/disable training ops at build time
          l2_penalty: float value, weight decay (l2) penalty

        Returns:
          Logits.
        """

        with tf.variable_scope(self.__class__.__name__):
            with tf.variable_scope("conv1"):
                conv1 = tf.nn.relu(
                    utils.batch_norm(
                        utils.conv_layer(
                            images, [5, 5, 1, 32],
                            1,
                            'SAME',
                            bias_term=False,
                            wd=l2_penalty), is_training_
                        if train_phase else False))

            with tf.variable_scope("pool1"):
                pool1 = tf.nn.max_pool(
                    conv1,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')

            with tf.variable_scope("conv2"):
                conv2 = tf.nn.relu(
                    utils.batch_norm(
                        utils.conv_layer(
                            pool1, [5, 5, 32, 64],
                            1,
                            'SAME',
                            bias_term=False,
                            wd=l2_penalty), is_training_
                        if train_phase else False))

            with tf.variable_scope("pool2"):
                pool2 = tf.nn.max_pool(
                    conv2,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')
                pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])

            with tf.variable_scope("fc1"):
                fc1 = tf.nn.relu(
                    utils.batch_norm(
                        utils.fc_layer(
                            pool2, [7 * 7 * 64, 1024],
                            bias_term=False,
                            wd=l2_penalty), is_training_
                        if train_phase else False))

            with tf.variable_scope("softmax_linear"):
                logits = utils.fc_layer(fc1, [1024, num_classes])
            return logits

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]

        Returns:
          Loss tensor of type float.
        """
        with tf.variable_scope('loss'):
            # Calculate the average cross entropy loss across the batch.
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(
                cross_entropy, name='cross_entropy')
            tf.add_to_collection(utils.LOSSES_COLLECTION, cross_entropy_mean)

            # The total loss is defined as the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            error = tf.add_n(
                tf.get_collection(utils.LOSSES_COLLECTION), name='total_loss')

        return error

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
            logits: the model output
        """
        is_training_ = tf.placeholder_with_default(
            False, shape=(), name="is_training_")
        # build a graph that computes the logits predictions from the images
        logits = self._inference(images, num_classes, is_training_, train_phase,
                                 l2_penalty)

        return is_training_, logits
