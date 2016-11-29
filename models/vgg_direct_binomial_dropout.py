#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Builds the VGG-like network with direct binomial dropout layers
applyed after avery layer of neurons"""

import tensorflow as tf
from . import utils


def inference(images,
              num_classes,
              is_training_,
              train_phase=False,
              l2_penalty=0.0):
    """Builds the VGG-like network with binomial dropout layers
    applyed after avery layer of neurons.

    Args:
      images: Images returned from distorted_inputs() or inputs().
      num_classes: Number of classes to predict
      is_training_: enable/disable training ops at run time
      train_phase: Boolean to enable/disable training ops at build time
      l2_penalty: float value, weight decay (l2) penalty

    Returns:
      Logits.
    """

    def binomial_direct_drop(layer, prob):
        """ Build a condition node if we are in train_phase. thus we can use the
        is_training_ placeholder to switch.
        Build a prob*layer node when we're not in train_phase.
        Returns the correct node"""

        def eval_phase():
            """ Scale activations with binomial prob at test time """
            # count the number of neurons
            num_neurons, _ = utils.num_neurons_and_shape(layer)
            # build the binomial distribution of num_neurons bernoullian experiments
            # each with probability prob
            dist = tf.contrib.distributions.Binomial(
                n=tf.cast(num_neurons, tf.float32), p=prob)
            # extract the P(dist = expected_kept_on) and scale the activations
            return (1.0 - dist.prob(num_neurons * prob)) * layer

        if train_phase:
            layer = tf.cond(
                tf.equal(is_training_, True),
                lambda: utils.direct_dropout(layer, prob), eval_phase)
        else:
            layer = eval_phase()
        return layer

    with tf.variable_scope('64'):
        with tf.variable_scope('conv1'):
            conv1 = tf.nn.relu(
                utils.conv_layer(
                    images, [3, 3, 3, 64], 1, 'SAME', wd=l2_penalty))
            binomial_direct_drop(conv1, 0.7)

        with tf.variable_scope('conv2'):
            conv2 = tf.nn.relu(
                utils.conv_layer(
                    conv1, [3, 3, 64, 64], 1, 'SAME', wd=l2_penalty))
            binomial_direct_drop(conv2, 0.6)

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(
            conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('128'):
        with tf.variable_scope('conv3'):
            conv3 = tf.nn.relu(
                utils.conv_layer(
                    pool1, [3, 3, 64, 128], 1, 'SAME', wd=l2_penalty))

            conv3 = binomial_direct_drop(conv3, 0.6)

        with tf.variable_scope('conv4'):
            conv4 = tf.nn.relu(
                utils.conv_layer(
                    conv3, [3, 3, 128, 128], 1, 'SAME', wd=l2_penalty))

            conv4 = binomial_direct_drop(conv4, 0.6)

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(
            conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('256'):
        with tf.variable_scope('conv5'):
            conv5 = tf.nn.relu(
                utils.conv_layer(
                    pool2, [3, 3, 128, 256], 1, 'SAME', wd=l2_penalty))

            conv5 = binomial_direct_drop(conv5, 0.6)

        with tf.variable_scope('conv6'):
            conv6 = tf.nn.relu(
                utils.conv_layer(
                    conv5, [3, 3, 256, 256], 1, 'SAME', wd=l2_penalty))

            conv6 = binomial_direct_drop(conv6, 0.6)

        with tf.variable_scope('conv7'):
            conv7 = tf.nn.relu(
                utils.conv_layer(
                    conv6, [3, 3, 256, 256], 1, 'SAME', wd=l2_penalty))

            conv7 = binomial_direct_drop(conv7, 0.6)

    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(
            conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('512'):
        with tf.variable_scope('conv8'):
            conv8 = tf.nn.relu(
                utils.conv_layer(
                    pool3, [3, 3, 256, 512], 1, 'SAME', wd=l2_penalty))

            conv8 = binomial_direct_drop(conv8, 0.6)

        with tf.variable_scope('conv9'):
            conv9 = tf.nn.relu(
                utils.conv_layer(
                    conv8, [3, 3, 512, 512], 1, 'SAME', wd=l2_penalty))

            conv9 = binomial_direct_drop(conv9, 0.6)

        with tf.variable_scope('conv10'):
            conv10 = tf.nn.relu(
                utils.conv_layer(
                    conv9, [3, 3, 512, 512], 1, 'SAME', wd=l2_penalty))

            conv10 = binomial_direct_drop(conv10, 0.6)

    with tf.variable_scope('pool4'):
        pool4 = tf.nn.max_pool(
            conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('512b2'):
        with tf.variable_scope('conv11'):
            conv11 = tf.nn.relu(
                utils.conv_layer(
                    pool4, [3, 3, 512, 512], 1, 'SAME', wd=l2_penalty))

            conv11 = binomial_direct_drop(conv11, 0.6)

        with tf.variable_scope('conv12'):
            conv12 = tf.nn.relu(
                utils.conv_layer(
                    conv11, [3, 3, 512, 512], 1, 'SAME', wd=l2_penalty))

            conv12 = binomial_direct_drop(conv12, 0.6)

        with tf.variable_scope('conv13'):
            conv13 = tf.nn.relu(
                utils.conv_layer(
                    conv12, [3, 3, 512, 512], 1, 'SAME', wd=l2_penalty))

            conv13 = binomial_direct_drop(conv13, 0.6)

    with tf.variable_scope('pool5'):
        pool5 = tf.nn.max_pool(
            conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        pool5 = tf.reshape(pool5, [-1, 512])

    with tf.variable_scope('fc'):
        fc1 = tf.nn.relu(utils.fc_layer(pool5, [512, 512], wd=l2_penalty))
        fc1 = binomial_direct_drop(fc1, 0.5)

    with tf.variable_scope('softmax_linear'):
        logits = utils.fc_layer(fc1, [512, num_classes], wd=0.0)
    return logits


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    error = tf.add_n(tf.get_collection('losses'), name='total_loss')
    utils.log(tf.summary.scalar('loss', error))
    return error


def get_model(images, num_classes, train_phase=False, l2_penalty=0.0):
    """ define the model with its inputs.
    Use this function to define the model in training and when exporting the model
    in the protobuf format.

    Args:
        images: model input
        num_classes: number of classes to predict
        train_phase: set it to True when defining the model, during train
        l2_penalty: float value, weight decay (l2) penalty

    Return:
        is_training_: tf.bool placeholder to enable/disable training ops at run time
        logits: the model output
    """
    is_training_ = tf.placeholder(tf.bool, shape=(), name="is_training_")
    # build a graph that computes the logits predictions from the images
    logits = inference(
        images,
        num_classes,
        is_training_,
        train_phase=train_phase,
        l2_penalty=l2_penalty)

    return is_training_, logits
