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
NAME = 'model2'

# Constants describing the training process.
BATCH_SIZE = 128
NUM_EPOCHS_PER_DECAY = 25  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
MOMENTUM = 0.9  # Momentum
WD_PENALTY = 5e-4  # L2(weights) penalty
INITIAL_LEARNING_RATE = 1e-2  # Initial learning rate.


def keep_prob_decay(validation_accuracy,
                    keep_prob,
                    min_keep_prob,
                    num_updates,
                    decay_amount,
                    name=None):
    """ Decay keep_prob until it reaches min_keep_prob.
    First calculates the trigger rule: decay keep prob only if
    trigger = floor(va/ (sum_{i=0}^{num_updates}{va_i} + va)/num_updates)
    that equals 0 if va is increasing, 1 if is constant or is decreasing.

    On every call to the function the trigger value is added to
    trigger_tot.
    This variable contains the number of decays to execute on the next step

    Then:
    keep_prob = max(min_keep_prob, keep_prob - decay_amount * trigger_top * trigger)

    Every validation accuracy update do this computation
    """

    with tf.name_scope(name, "KeepProbDecay", [
            validation_accuracy, keep_prob, min_keep_prob, num_updates,
            decay_amount
    ]) as name:
        validation_accuracy = tf.convert_to_tensor(
            validation_accuracy, name="validation_accuracy", dtype=tf.float32)
        keep_prob = tf.convert_to_tensor(
            keep_prob, name="keep_prob", dtype=tf.float32)
        min_keep_prob = tf.convert_to_tensor(
            min_keep_prob, name="min_keep_prob", dtype=tf.float32)
        decay_amount = tf.convert_to_tensor(
            decay_amount, name="decay_amount", dtype=tf.float32)

        # crate a tensor with num_updates value, to accumulte validation accuracies
        accumulator = tf.Variable(
            tf.zeros(
                [num_updates], dtype=tf.float32), trainable=False)
        position = tf.Variable(0, dtype=tf.int32, trainable=False)
        accumulated = tf.Variable(0, dtype=tf.float32, trainable=False)
        trigger_tot = tf.Variable(0, dtype=tf.float32, trainable=False)

        # convert num_updates to a tensor
        num_updates = tf.convert_to_tensor(
            num_updates, name="num_updates", dtype=tf.float32)

        # when validation accuracy gets updated
        on_value_change = tf.identity([validation_accuracy])
        with tf.control_dependencies([on_value_change]):
            # calculate right position in the accumulator vector
            # where we put the va value
            tf.assign(position,
                      tf.cast(tf.mod(accumulated, num_updates), tf.int32))
            # update value
            tf.scatter_update(accumulator, position, validation_accuracy)
            # update the amount of accumulated value of the whole train process
            tf.assign_add(accumulated, 1)

            # get the denominator
            denominator = tf.cond(
                tf.greater_equal(accumulated, num_updates),
                lambda: num_updates, lambda: accumulated)

            # calculate cumulative rolling average
            rolling_avg = tf.reduce_sum(accumulator) / denominator

            def firstIteration():
                return keep_prob

            def otherIteration():
                # trigger value
                trigger = tf.floor(0.5 + (validation_accuracy / rolling_avg))
                # sum number of triggered decays
                tf.assign_add(trigger_tot, trigger)
                new_keep_prob = tf.maximum(
                    min_keep_prob,
                    keep_prob - decay_amount * trigger_tot * trigger)

                utils.log(
                    tf.scalar_summary([
                        'decay_amount', 'trigger_tot', 'trigger'
                    ], [decay_amount, trigger_tot, trigger]))
                return new_keep_prob

            new_keep_prob = tf.cond(
                tf.equal(rolling_avg, 0.0), firstIteration, otherIteration)
            utils.log(tf.scalar_summary('keep_prob', new_keep_prob))
            return new_keep_prob


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
                conv1 = tf.nn.dropout(conv1, keep_prob)

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
                conv3 = tf.nn.dropout(conv3, keep_prob)

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
                conv5 = tf.nn.dropout(conv5, keep_prob)

        with tf.variable_scope('conv6'):
            conv6 = tf.nn.relu(
                utils.conv_layer(
                    conv5, [3, 3, 256, 256], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv6 = tf.nn.dropout(conv6, keep_prob)
                #conv6 = tf.nn.dropout(conv6, 1 - 0.4)
                # removed
                pass

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
                conv8 = tf.nn.dropout(conv8, keep_prob)
                #conv8 = tf.nn.dropout(conv8, 1 - 0.4)

        with tf.variable_scope('conv9'):
            conv9 = tf.nn.relu(
                utils.conv_layer(
                    conv8, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv9 = tf.nn.dropout(conv9, keep_prob)
                #conv9 = tf.nn.dropout(conv9, 1 - 0.4)
                # removed
                pass

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
                conv11 = tf.nn.dropout(conv11, keep_prob)
                #conv11 = tf.nn.dropout(conv11, 1 - 0.4)

        with tf.variable_scope('conv12'):
            conv12 = tf.nn.relu(
                utils.conv_layer(
                    conv11, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv12 = tf.nn.dropout(conv12, keep_prob)
                #conv12 = tf.nn.dropout(conv12, 1 - 0.4)
                # removed
                pass

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
            #pool5 = tf.nn.dropout(pool5, 0.5)
            # removed
            pass

        pool5 = tf.reshape(pool5, [-1, 512])

    with tf.variable_scope('fc'):
        fc1 = tf.nn.relu(utils.fc_layer(pool5, [512, 512], wd=WD_PENALTY))

        if train_phase:
            fc1 = tf.nn.dropout(fc1, keep_prob)
            #fc1 = tf.nn.dropout(fc1, 0.5)

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
