#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utils for models creation"""

import numbers
import math
import tensorflow as tf

# name of the collection that holds non trainable
# but required variables for the current model
REQUIRED_NON_TRAINABLES = 'required_vars_collection'

# name of the collection that holds the summaries
# related to the model (and the train phase)
MODEL_SUMMARIES = 'model_summaries'

# losses collection
LOSSES_COLLECTION = 'losses'


def tf_log(summary, collection=MODEL_SUMMARIES):
    """Add tf.summary object to collection named collection"""
    tf.add_to_collection(collection, summary)


# Adapeted from
# https://gist.github.com/kukuruza/03731dc494603ceab0c5#gistcomment-1879326
def put_kernels_on_grid(kernel, grid_side, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
        kernel:    tensor of shape [Y, X, NumChannels, NumKernels]
        grid_side: side of the grid. Require: NumKernels == grid_side**2
        pad:       number of black pixels around each filter (between them)

    Returns:
        An image Tensor with shape [(Y+2*pad)*grid_side, (X+2*pad)*grid_side, NumChannels, 1].
    """

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(
        kernel1,
        tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]),
        mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2,
                    tf.stack(
                        values=[grid_side, Y * grid_side, X, channels],
                        axis=0))  #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4,
                    tf.stack(
                        values=[1, X * grid_side, Y * grid_side, channels],
                        axis=0))  #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def weight(name,
           shape,
           initializer=tf.contrib.layers.variance_scaling_initializer(
               factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32),
           wd=0.0):
    """Returns a tensor with the requested shape, initialized
      using the provided intitializer (default: He init).
      Applies L2 weight decay penalyt using wd term"""
    weights = tf.get_variable(
        name, shape=shape, initializer=initializer, dtype=tf.float32)

    # show weights of the first layer
    first_layer = len(shape) == 4 and shape[2] in (1, 3, 4)
    if first_layer:
        num_kernels = shape[3]
        # check if is a perfect square
        grid_side = math.floor(math.sqrt(num_kernels))
        tf_log(
            tf.summary.image(name,
                             put_kernels_on_grid(weights[:, :, :, 0:grid_side**
                                                         2], grid_side,
                                                 grid_side)))

    # Add weight decay to W
    tf.add_to_collection(LOSSES_COLLECTION,
                         tf.multiply(tf.nn.l2_loss(weights), wd))
    tf_log(tf.summary.histogram(name, weights))
    return weights


def bias(name, shape, initializer=tf.constant_initializer(value=0.0)):
    """Returns a bias variabile initializeted wuth the provided initializer.
    No weight decay applied to the bias terms"""
    return weight(name, shape, initializer, wd=0.0)


def atrous_conv_layer(input_x,
                      shape,
                      rate,
                      padding,
                      bias_term=True,
                      activation=tf.identity,
                      wd=0.0):
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
         bias_term: a boolean to add (if True) the bias term. Usually disable when
             the layer is wrapped in a batch norm layer
         activation: activation function. Default linear
         wd: weight decay
    Rerturns the conv2d op"""

    W = weight("W", shape, wd=wd)
    result = tf.nn.atrous_conv2d(input_x, W, rate, padding)
    if bias_term:
        b = bias("b", [shape[3]])
        result = tf.nn.bias_add(result, b)

    # apply nonlinearity
    out = activation(result)

    # log convolution result pre-activation function
    # on a single image, the first of the batch
    conv_results = tf.split(
        value=result[0], num_or_size_splits=shape[3], axis=2)
    grid_side = math.floor(math.sqrt(shape[3]))

    pre_activation = put_kernels_on_grid(
        tf.transpose(conv_results, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
        grid_side, grid_side)

    # log post-activation
    conv_results = tf.split(value=out[0], num_or_size_splits=shape[3], axis=2)
    post_activation = put_kernels_on_grid(
        tf.transpose(conv_results, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
        grid_side, grid_side)

    tf_log(
        tf.summary.image(
            result.name + '/pre_post_activation',
            tf.concat([pre_activation, post_activation], axis=2),
            max_outputs=1))
    return out


def conv_layer(input_x,
               shape,
               stride,
               padding,
               bias_term=True,
               activation=tf.identity,
               wd=0.0):
    """ Define a conv layer.
    Args:
         input_x: a 4D tensor
         shape: weight shape
         stride: a single value supposing equal stride along X and Y
         padding: 'VALID' or 'SAME'
         bias_term: a boolean to add (if True) the bias term. Usually disable when
             the layer is wrapped in a batch norm layer
         activation: activation function. Default linear
         wd: weight decay
    Rerturns the conv2d op"""

    W = weight("W", shape, wd=wd)
    result = tf.nn.conv2d(input_x, W, [1, stride, stride, 1], padding)
    if bias_term:
        b = bias("b", [shape[3]])
        result = tf.nn.bias_add(result, b)

    # apply nonlinearity
    out = activation(result)

    # log convolution result pre-activation function
    # on a single image, the first of the batch
    conv_results = tf.split(
        value=result[0], num_or_size_splits=shape[3], axis=2)
    grid_side = math.floor(math.sqrt(shape[3]))

    pre_activation = put_kernels_on_grid(
        tf.transpose(conv_results, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
        grid_side, grid_side)

    # log post-activation
    conv_results = tf.split(value=out[0], num_or_size_splits=shape[3], axis=2)
    post_activation = put_kernels_on_grid(
        tf.transpose(conv_results, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
        grid_side, grid_side)

    tf_log(
        tf.summary.image(
            result.name + '/pre_post_activation',
            tf.concat([pre_activation, post_activation], axis=2),
            max_outputs=1))
    return out


def fc_layer(input_x, shape, bias_term=True, activation=tf.identity, wd=0.0):
    """ Define a fully connected layer.
    Args:
        input_x: a 4d tensor
        shape: weight shape
        bias_term: a boolean to add (if True) the bias term. Usually disable when
             the layer is wrapped in a batch norm layer
        activation: activation function. Default linear
        wd: weight decay
    Returns the fc layer"""

    W = weight("W", shape, wd=wd)
    result = tf.matmul(input_x, W)
    if bias_term:
        b = bias("b", [shape[1]])
        result = tf.nn.bias_add(result, b)

    return activation(result)


def batch_norm(layer_output, is_training_, decay=0.9):
    """Applies batch normalization to the layer output.
    Args:
        layer_output: 4-d tensor, output of a FC/convolutional layer
        is_training_: placeholder or boolean variable to set to True when training
        decay:        decay for the moving average.
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


def variables_to_save(add_list=[]):
    """Returns a list of variables to save.
    add_list variables are always added to the list
    """
    return tf.trainable_variables() + tf.get_collection_ref(
        REQUIRED_NON_TRAINABLES) + add_list


def variables_to_restore(add_list=[], exclude_scope_list=[]):
    """Returns a list of variables to restore to made the model working
    properly.
    The list is made by the trainable variables + required non trainable variables
    such as statistics of batch norm layers.
    Remove from the list variables that are in the exclude_scope_list.
    Add variables in the add_list
    """

    variables = variables_to_save()
    if len(exclude_scope_list) > 0:
        variables[:] = [
            variable for variable in variables
            if not variable.name.startswith(
                tuple(scope for scope in exclude_scope_list))
        ]
    return variables + add_list


def variables_to_train(scope_list=[]):
    """Returns a list of variables to train, filtered by the scopes.
    Returns:
        the list of variables to train by the optimizer
    """
    if len(scope_list) == 0:
        return tf.trainable_variables()
    variables_to_train = []
    for scope in scope_list:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def num_neurons_and_shape(layer):
    """Count the number of neurons in a single element of the layer, returns this
    number and the shape of the single layer.
    Args:
        layer: [batch_size, widht, height, depth] if the layer is convolutional
               [batch_size, num_neruons] if the layer is fully connected
    Returns:
        num_neurons, shape
        Where num_neurons is the number of neurons in a single elment of the input batch,
        shape is the shape of the single element"""
    # extract the number of neurons in x
    # and the number of neurons kept on
    input_shape = layer.get_shape()
    if len(input_shape) == 4:  # conv layer
        num_neurons = input_shape[1].value * input_shape[2].value * input_shape[
            3].value
        shape = [
            -1, input_shape[1].value, input_shape[2].value, input_shape[3].value
        ]
    else:  #fc layer
        num_neurons = input_shape[1].value
        shape = [-1, input_shape[1].value]

    return num_neurons, shape


def active_neurons(layer, off_value=0):
    """Count the number of active (> off_value) neurons in a single element of the layer.
    Args:
        layer: [batch_size, widht, height, depth] if the layer is convolutional
               [batch_size, num_neruons] if the layer is fully connected
    Returns:
        kept_on: [batch_size, 1] tf.int32, number of active neurons
    """
    binary_tensor = tf.cast(tf.greater(layer, off_value), tf.int32)
    return tf.reduce_sum(binary_tensor, [1, 2, 3]
                         if len(layer.get_shape()) == 4 else [1])


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
    with tf.name_scope(name, "direct_dropout", [x]) as name:
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
