#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utils for models creation"""

import numbers
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


def binomial_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """Computes dropout.
    With probability `keep_prob`, outputs the input element scaled up by
    `1 / P(Binomial(num_neurons(x), keep_prob) = num_neurons(x)*keep_prob)`,
    otherwise outputs `0`. The scaling is so that the expected sum is unchanged.
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.
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
    with tf.name_scope(name, "binomial_dropout", [x]) as name:
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

        # scale using the probability of dropping np neurons
        # from a binomial distribution

        # extract the number of neurons in x
        input_shape = x.get_shape()
        if len(input_shape) == 4:  # conv layer
            num_neurons = input_shape[1].value * input_shape[
                2].value * input_shape[3].value
        else:  #fc layer
            num_neurons = input_shape[1].value

        num_neurons = tf.convert_to_tensor(
            num_neurons, dtype=tf.float32, name="num_neurons")
        dist = tf.contrib.distributions.Binomial(n=num_neurons, p=keep_prob)
        expected_kept_on = num_neurons * keep_prob
        prob = dist.prob(expected_kept_on)

        def drop():
            ret = tf.div(x, 1 - prob) * binary_tensor
            ret.set_shape(x.get_shape())
            return ret

        return tf.cond(tf.equal(keep_prob, 1.0), lambda: x, drop)
