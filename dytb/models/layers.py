#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Layers to easy create models and nice visualizations"""

import math
import numbers
import tensorflow as tf
from .utils import legalize_name, tf_log
from .visualization import on_grid
from .collections import LOSSES, REQUIRED_NON_TRAINABLES


def weight(name,
           shape,
           train_phase,
           initializer=tf.contrib.layers.variance_scaling_initializer(
               factor=2.0,
               mode='FAN_IN',
               uniform=False,
               seed=None,
               dtype=tf.float32),
           wd=0.0):
    """Returns a tensor with the requested shape, initialized
      using the provided intitializer (default: He init).
      Applies L2 weight decay penalyt using wd term.
      Enables visualizations when in train_phase=True.
      Args:
          name: the name of the weight
          shape: the shape of the tensor, a python list or tuple
          train_phase: boolean, to enable/disable visualization and L2 decay
          initializer: the initializer to use
          wd: when train_phase=True uses `wd` as decay penalty
      Returns:
        weights: the weight tensor initialized.
    """
    weights = tf.get_variable(
        name, shape=shape, initializer=initializer, dtype=tf.float32)

    # show weights of the first layer
    first = len(shape) == 4 and shape[2] in (1, 3, 4)
    if first and train_phase:
        num_kernels = shape[3]
        # check if is a perfect square
        grid_side = math.floor(math.sqrt(num_kernels))
        tf_log(
            tf.summary.image(
                legalize_name(name),
                on_grid(weights[:, :, :, 0:grid_side**2], grid_side,
                        grid_side)))

    if train_phase:
        # Add weight decay to W
        tf.add_to_collection(LOSSES, tf.multiply(tf.nn.l2_loss(weights), wd))
        tf_log(tf.summary.histogram(legalize_name(name), weights))
    return weights


def bias(name,
         shape,
         train_phase,
         initializer=tf.constant_initializer(value=0.0)):
    """Returns a bias variabile initializeted wuth the provided initializer.
    No weight decay applied to the bias terms.
    Enables visualization when in train_phase.
    Args:
        name: name for the bias variable
        shape: shape of the variabile
        train_phase: boolean, to enable/disable visualization
        initializer: the initializer to use
    Returns:
        bias: the vias variable correctly initialized
    """
    return weight(name, shape, train_phase, initializer=initializer, wd=0.0)


def atrous_conv(input_x,
                shape,
                rate,
                padding,
                train_phase,
                bias_term=True,
                activation=tf.identity,
                wd=0.0,
                initializer=tf.contrib.layers.variance_scaling_initializer(
                    factor=2.0,
                    mode='FAN_IN',
                    uniform=False,
                    seed=None,
                    dtype=tf.float32)):
    """ Define an atrous conv layer.
    Args:
         input_x: a 4D tensor
         shape: weight shape
         rate: : A positive int32. The stride with which we sample input values
            cross the height and width dimensions. Equivalently, the rate by which
            we upsample the filter values by inserting zeros
            across the height and width dimensions. In the literature, the same
            parameter is sometimes called input stride or dilation
         padding: 'VALID' or 'SAME'
         train_phase: boolean that enables/diables visualizations and train-only specific ops
         bias_term: a boolean to add (if True) the bias term. Usually disable when
             the layer is wrapped in a batch norm layer
         activation: activation function. Default linear
         wd: weight decay
         initializer: the initializer to use
    Rerturns:
        op: the conv2d op"""

    W = weight("W", shape, train_phase, initializer=initializer, wd=wd)
    result = tf.nn.atrous_conv2d(input_x, W, rate, padding)
    if bias_term:
        b = bias("b", [shape[3]], train_phase)
        result = tf.nn.bias_add(result, b)

    # apply nonlinearity
    out = activation(result)

    if train_phase:
        # log convolution result pre-activation function
        # on a single image, the first of the batch
        conv_results = tf.split(
            value=result[0], num_or_size_splits=shape[3], axis=2)
        grid_side = math.floor(math.sqrt(shape[3]))

        pre_activation = on_grid(
            tf.transpose(conv_results,
                         perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2], grid_side,
            grid_side)

        # log post-activation
        conv_results = tf.split(
            value=out[0], num_or_size_splits=shape[3], axis=2)
        post_activation = on_grid(
            tf.transpose(conv_results,
                         perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2], grid_side,
            grid_side)

        tf_log(
            tf.summary.image(
                legalize_name(result.name + '/pre_post_activation'),
                tf.concat([pre_activation, post_activation], axis=2),
                max_outputs=1))
    return out


def conv(input_x,
         shape,
         stride,
         padding,
         train_phase,
         bias_term=True,
         activation=tf.identity,
         wd=0.0,
         initializer=tf.contrib.layers.variance_scaling_initializer(
             factor=2.0,
             mode='FAN_IN',
             uniform=False,
             seed=None,
             dtype=tf.float32)):
    """ Define a conv layer.
    Args:
        input_x: a 4D tensor
        shape: weight shape
        stride: a single value supposing equal stride along X and Y
        padding: 'VALID' or 'SAME'
        train_phase: boolean that enables/diables visualizations and train-only specific ops
        bias_term: a boolean to add (if True) the bias term. Usually disable when
                   the layer is wrapped in a batch norm layer
        activation: activation function. Default linear
        train_phase: boolean that enables/diables visualizations and train-only specific ops
        wd: weight decay
        initializer: the initializer to use
    Rerturns:
        op: the conv2d op
    """

    W = weight("W", shape, train_phase, initializer=initializer, wd=wd)
    result = tf.nn.conv2d(input_x, W, [1, stride, stride, 1], padding)
    if bias_term:
        b = bias("b", [shape[3]], train_phase)
        result = tf.nn.bias_add(result, b)

    # apply nonlinearity
    out = activation(result)

    if train_phase:
        # log convolution result pre-activation function
        # on a single image, the first of the batch
        conv_results = tf.split(
            value=result[0], num_or_size_splits=shape[3], axis=2)
        grid_side = math.floor(math.sqrt(shape[3]))

        pre_activation = on_grid(
            tf.transpose(conv_results,
                         perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2], grid_side,
            grid_side)

        # log post-activation
        conv_results = tf.split(
            value=out[0], num_or_size_splits=shape[3], axis=2)
        post_activation = on_grid(
            tf.transpose(conv_results,
                         perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2], grid_side,
            grid_side)

        tf_log(
            tf.summary.image(
                legalize_name(result.name + '/pre_post_activation'),
                tf.concat([pre_activation, post_activation], axis=2),
                max_outputs=1))
    return out


def fc(input_x,
       shape,
       train_phase,
       bias_term=True,
       activation=tf.identity,
       wd=0.0,
       initializer=tf.contrib.layers.variance_scaling_initializer(
           factor=2.0,
           mode='FAN_IN',
           uniform=False,
           seed=None,
           dtype=tf.float32)):
    """ Define a fully connected layer.
    Args:
        input_x: a 4d tensor
        shape: weight shape
        train_phase: boolean that enables/diables visualizations and train-only specific ops
        bias_term: a boolean to add (if True) the bias term. Usually disable when
             the layer is wrapped in a batch norm layer
        activation: activation function. Default linear
        wd: weight decay
        initializer: the initializer to use
    Returns:
        fc: the fc layer
    """

    W = weight("W", shape, train_phase, initializer=initializer, wd=wd)
    result = tf.matmul(input_x, W)
    if bias_term:
        b = bias("b", [shape[1]], train_phase)
        result = tf.nn.bias_add(result, b)

    return activation(result)


def batch_norm(layer_output, is_training_, decay=0.9):
    """Applies batch normalization to the layer output.
    Args:
        layer_output: 4-d tensor, output of a FC/convolutional layer
        is_training_: placeholder or boolean variable to set to True when training
        decay:        decay for the moving average.
    Returns:
        bn: the batch normalization layer
    """
    return tf.contrib.layers.batch_norm(
        inputs=layer_output,
        decay=decay,
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
        batch_weights=None,
        fused=True,
        scope=None)


def direct_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """Computes dropout.
    The original dropout as described in the paper, not the inverted version.
    Thus it requires to scale the activation AT TEST TIME.
    Args:
        x: A tensor.
        keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
        noise_shape: A 1-D `Tensor` of type `int32`, representing the
          shape for randomly generated keep/drop flags.
        seed: A Python integer. Used to create random seeds.
        name: A name for this operation (optional).
    Returns:
        A Tensor of the same shape of `x`.
    Raises:
        ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    with tf.name_scope(name, "direct_dropout", [x]):
        x = tf.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError(
                "keep_prob must be a scalar tensor or a float in the "
                "range (0, 1], got %g" % keep_prob)
        keep_prob = tf.convert_to_tensor(
            keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tf.TensorShape([]))

        # Do nothing if we know keep_prob == 1
        if tf.contrib.util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else tf.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(
            noise_shape, seed=seed, dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        # Do not scale the activation in train time
        ret = tf.multiply(x, binary_tensor)
        ret.set_shape(x.get_shape())
        return ret
