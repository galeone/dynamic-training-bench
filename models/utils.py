#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utils for models creation"""

import tensorflow as tf

# name of the collection that holds non trainable
# but required variables for the current model
REQUIRED_NON_TRAINABLES = 'required_vars_collection'


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


def batch_norm(layer_output, is_training_):
    """Applies batch normalization to the layer output.
    Args:
        layer_output: 4-d tensor, output of a FC/convolutional layer
        is_training_: placeholder or boolean variable to set to True when training
    """
    return tf.contrib.layers.batch_norm(
        layer_output,
        decay=0.999,
        center=True,
        scale=True,
        epsilon=1e-3,
        activation_fn=None,
        # update moving mean and variance in place
        updates_collections=None,
        is_training=is_training_,
        reuse=None,
        # create a collections of varialbes to save
        # (moving mean and moving variance)
        variables_collections=[REQUIRED_NON_TRAINABLES],
        outputs_collections=None,
        trainable=True,
        scope=None)


def log(summary):
    """Add summary to train_summaries collection"""
    tf.add_to_collection('train_summaries', summary)


def variables_to_save(addlist):
    """Create a list of all trained variables and required variables of the model.
    Appends to the list, the addlist passed as argument.

    Args:
        addlist: (list, of, variables, to, save)
    Returns:
        a a list of variables"""

    return tf.trainable_variables() + tf.get_collection_ref(
        REQUIRED_NON_TRAINABLES) + addlist
