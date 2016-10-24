#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Builds the VGG-like network."""

import tensorflow as tf
from inputs import cifar10 as dataset
from . import utils

# Model name
NAME = 'model1'

# Constants describing the training process.
BATCH_SIZE = 128
NUM_EPOCHS_PER_DECAY = 25  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
MOMENTUM = 0.9  # Momentum
WD_PENALTY = 5e-4  # L2(weights) penalty
INITIAL_LEARNING_RATE = 1e-2  # Initial learning rate.


def inference(images, keep_prob, train_phase=False):
    """Build the CIFAR-10 VGG model.

  Args:
    images: Images returned from distorted_inputs() or inputs().
    keep_prob: tensor for the dropout probability of keep neurons active

  Returns:
    Logits.
  """
    with tf.variable_scope('64'):
        with tf.variable_scope('conv1'):
            conv1 = tf.nn.relu(
                utils.conv_layer(
                    images, [3, 3, 3, 64], 1, 'SAME', wd=WD_PENALTY))
            if train_phase:
                #conv1 = tf.nn.dropout(conv1, keep_prob)
                conv1 = tf.nn.dropout(conv1, 1 - 0.3)

        with tf.variable_scope('conv2'):
            conv2 = tf.nn.relu(
                utils.conv_layer(
                    conv1, [3, 3, 64, 64], 1, 'SAME', wd=WD_PENALTY))

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(
            conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('128'):
        with tf.variable_scope('conv3'):
            conv3 = tf.nn.relu(
                utils.conv_layer(
                    pool1, [3, 3, 64, 128], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv3 = tf.nn.dropout(conv3, keep_prob)
                conv3 = tf.nn.dropout(conv3, 1 - 0.4)

        with tf.variable_scope('conv4'):
            conv4 = tf.nn.relu(
                utils.conv_layer(
                    conv3, [3, 3, 128, 128], 1, 'SAME', wd=WD_PENALTY))

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(
            conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('256'):
        with tf.variable_scope('conv5'):
            conv5 = tf.nn.relu(
                utils.conv_layer(
                    pool2, [3, 3, 128, 256], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv5 = tf.nn.dropout(conv5, keep_prob)
                conv5 = tf.nn.dropout(conv5, 1 - 0.4)

        with tf.variable_scope('conv6'):
            conv6 = tf.nn.relu(
                utils.conv_layer(
                    conv5, [3, 3, 256, 256], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv6 = tf.nn.dropout(conv6, keep_prob)
                conv6 = tf.nn.dropout(conv6, 1 - 0.4)

        with tf.variable_scope('conv7'):
            conv7 = tf.nn.relu(
                utils.conv_layer(
                    conv6, [3, 3, 256, 256], 1, 'SAME', wd=WD_PENALTY))

    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(
            conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('512'):
        with tf.variable_scope('conv8'):
            conv8 = tf.nn.relu(
                utils.conv_layer(
                    pool3, [3, 3, 256, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv8 = tf.nn.dropout(conv8, keep_prob)
                conv8 = tf.nn.dropout(conv8, 1 - 0.4)

        with tf.variable_scope('conv9'):
            conv9 = tf.nn.relu(
                utils.conv_layer(
                    conv8, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv9 = tf.nn.dropout(conv9, keep_prob)
                conv9 = tf.nn.dropout(conv9, 1 - 0.4)

        with tf.variable_scope('conv10'):
            conv10 = tf.nn.relu(
                utils.conv_layer(
                    conv9, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

    with tf.variable_scope('pool4'):
        pool4 = tf.nn.max_pool(
            conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('512b2'):
        with tf.variable_scope('conv11'):
            conv11 = tf.nn.relu(
                utils.conv_layer(
                    pool4, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv11 = tf.nn.dropout(conv11, keep_prob)
                conv11 = tf.nn.dropout(conv11, 1 - 0.4)

        with tf.variable_scope('conv12'):
            conv12 = tf.nn.relu(
                utils.conv_layer(
                    conv11, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv12 = tf.nn.dropout(conv12, keep_prob)
                conv12 = tf.nn.dropout(conv12, 1 - 0.4)

        with tf.variable_scope('conv13'):
            conv13 = tf.nn.relu(
                utils.conv_layer(
                    conv12, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

    with tf.variable_scope('pool5'):
        pool5 = tf.nn.max_pool(
            conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # dropout on max-pooling, mfw
        if train_phase:
            #pool5 = tf.nn.dropout(pool5, keep_prob)
            pool5 = tf.nn.dropout(pool5, 0.5)

        pool5 = tf.reshape(pool5, [-1, 512])

    with tf.variable_scope('fc'):
        fc1 = tf.nn.relu(utils.fc_layer(pool5, [512, 512], wd=WD_PENALTY))

        if train_phase:
            #fc1 = tf.nn.dropout(fc1, keep_prob)
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('softmax_linear'):
        logits = utils.fc_layer(fc1, [512, dataset.NUM_CLASSES], wd=WD_PENALTY)
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
    utils.log(tf.scalar_summary('loss', error))
    return error


def train(total_loss, global_step):
    """Train model.
    Create an optimizer and apply to all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)

    utils.log(tf.scalar_summary('learning_rate', learning_rate))
    opt = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    train_op = opt.minimize(total_loss, global_step=global_step)

    return train_op


def get_model(images, train_phase=False):
    """ define the model with its inputs.
    Use this function to define the model in training and when exporting the model
    in the protobuf format.

    Args:
        images: model input
        train_phase: set it to True when defining the model, during train

    Return:
        keep_prob_: model dropout placeholder
        logits: the model output
    """
    keep_prob_ = tf.placeholder(tf.float32, shape=(), name="keep_prob_")
    # build a graph that computes the logits predictions from the images
    logits = inference(images, keep_prob_, train_phase=train_phase)

    return keep_prob_, logits
