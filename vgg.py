# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Builds the VGG-like network."""

import tensorflow as tf
from inputs import cifar10

# Constants describing the training process.
BATCH_SIZE = 128
NUM_EPOCHS_PER_DECAY = 25  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
MOMENTUM = 0.9  # Momentum
WD_PENALTY = 5e-4  # L2(weights) penalty
INITIAL_LEARNING_RATE = 1e-2  # Initial learning rate.


def _activation_summary(x):
    """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def weight(name,
           shape,
           initializer=tf.contrib.layers.variance_scaling_initializer(
               factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)):
    """ weight returns a tensor with the requested shape, initialized
      using the provided intitializer (default: He init)."""
    return tf.get_variable(
        name, shape=shape, initializer=initializer, dtype=tf.float32)


def bias(name, shape, initializer=tf.constant_initializer(value=0.0)):
    """Returns a bias variabile initializeted wuth the provided initializer"""
    return weight(name, shape, initializer)


def conv_layer(input_x, shape, stride, padding, wd=0.0):
    """ Define a conv layer.
    Args:
         input_x: a 4D tensor
         shape: weight shape
         stride: a single value supposing equal stride along X and Y
         padding: 'VALID' or 'SAME'
         wd: weight decay
    Rerturns the conv2d op"""
    W = weight("W", shape)
    b = bias("b", [shape[3]])
    # Add weight decay to W
    weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return tf.nn.bias_add(
        tf.nn.conv2d(input_x, W, [1, stride, stride, 1], padding), b)


def fc_layer(input_x, shape, wd=0.0):
    """ Define a fully connected layer.
    Args:
        input_x: a 4d tensor
        shape: weight shape
        wd: weight decay
    Returns the fc layer"""
    W = weight("W", shape)
    b = bias("b", [shape[1]])
    # Add weight decay to W
    weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return tf.nn.bias_add(tf.matmul(input_x, W), b)


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
                conv_layer(
                    images, [3, 3, 3, 64], 1, 'SAME', wd=WD_PENALTY))
            if train_phase:
                #conv1 = tf.nn.dropout(conv1, keep_prob)
                conv1 = tf.nn.dropout(conv1, 1 - 0.3)
            _activation_summary(conv1)

        with tf.variable_scope('conv2'):
            conv2 = tf.nn.relu(
                conv_layer(
                    conv1, [3, 3, 64, 64], 1, 'SAME', wd=WD_PENALTY))
            _activation_summary(conv2)

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(
            conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('128'):
        with tf.variable_scope('conv3'):
            conv3 = tf.nn.relu(
                conv_layer(
                    pool1, [3, 3, 64, 128], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv3 = tf.nn.dropout(conv3, keep_prob)
                conv3 = tf.nn.dropout(conv3, 1 - 0.4)
            _activation_summary(conv3)

        with tf.variable_scope('conv4'):
            conv4 = tf.nn.relu(
                conv_layer(
                    conv3, [3, 3, 128, 128], 1, 'SAME', wd=WD_PENALTY))
            _activation_summary(conv4)

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(
            conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('256'):
        with tf.variable_scope('conv5'):
            conv5 = tf.nn.relu(
                conv_layer(
                    pool2, [3, 3, 128, 256], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv5 = tf.nn.dropout(conv5, keep_prob)
                conv5 = tf.nn.dropout(conv5, 1 - 0.4)
            _activation_summary(conv5)

        with tf.variable_scope('conv6'):
            conv6 = tf.nn.relu(
                conv_layer(
                    conv5, [3, 3, 256, 256], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv6 = tf.nn.dropout(conv6, keep_prob)
                conv6 = tf.nn.dropout(conv6, 1 - 0.4)
            _activation_summary(conv6)

        with tf.variable_scope('conv7'):
            conv7 = tf.nn.relu(
                conv_layer(
                    conv6, [3, 3, 256, 256], 1, 'SAME', wd=WD_PENALTY))
            _activation_summary(conv7)

    with tf.variable_scope('pool3'):
        pool3 = tf.nn.max_pool(
            conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('512'):
        with tf.variable_scope('conv8'):
            conv8 = tf.nn.relu(
                conv_layer(
                    pool3, [3, 3, 256, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv8 = tf.nn.dropout(conv8, keep_prob)
                conv8 = tf.nn.dropout(conv8, 1 - 0.4)
            _activation_summary(conv8)

        with tf.variable_scope('conv9'):
            conv9 = tf.nn.relu(
                conv_layer(
                    conv8, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv9 = tf.nn.dropout(conv9, keep_prob)
                conv9 = tf.nn.dropout(conv9, 1 - 0.4)
            _activation_summary(conv9)

        with tf.variable_scope('conv10'):
            conv10 = tf.nn.relu(
                conv_layer(
                    conv9, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))
            _activation_summary(conv10)

    with tf.variable_scope('pool4'):
        pool4 = tf.nn.max_pool(
            conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('512b2'):
        with tf.variable_scope('conv11'):
            conv11 = tf.nn.relu(
                conv_layer(
                    pool4, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv11 = tf.nn.dropout(conv11, keep_prob)
                conv11 = tf.nn.dropout(conv11, 1 - 0.4)
            _activation_summary(conv11)

        with tf.variable_scope('conv12'):
            conv12 = tf.nn.relu(
                conv_layer(
                    conv11, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))

            if train_phase:
                #conv12 = tf.nn.dropout(conv12, keep_prob)
                conv12 = tf.nn.dropout(conv12, 1 - 0.4)
            _activation_summary(conv12)

        with tf.variable_scope('conv13'):
            conv13 = tf.nn.relu(
                conv_layer(
                    conv12, [3, 3, 512, 512], 1, 'SAME', wd=WD_PENALTY))
            _activation_summary(conv13)

    with tf.variable_scope('pool5'):
        pool5 = tf.nn.max_pool(
            conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # dropout on max-pooling, mfw
        #pool5 = tf.nn.dropout(pool5, keep_prob)
        pool5 = tf.nn.dropout(pool5, 0.5)

        pool5 = tf.reshape(pool5, [-1, 512])

    with tf.variable_scope('fc'):
        fc1 = tf.nn.relu(fc_layer(pool5, [512, 512], wd=WD_PENALTY))

        if train_phase:
            #fc1 = tf.nn.dropout(fc1, keep_prob)
            fc1 = tf.nn.dropout(fc1, 0.5)
        _activation_summary(fc1)

    with tf.variable_scope('softmax_linear'):
        logits = fc_layer(fc1, [512, cifar10.NUM_CLASSES], wd=WD_PENALTY)
        _activation_summary(logits)
    return logits


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
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
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
    # Variables that affect learning rate.
    num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, MOMENTUM)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


def get_model(images, train_phase):
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
