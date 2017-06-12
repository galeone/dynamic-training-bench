#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""utility methods to create visualizations in tensorboard"""

import math
import tensorflow as tf
from .utils import tf_log


# Adapeted from
# https://gist.github.com/kukuruza/03731dc494603ceab0c5#gistcomment-1879326
def on_grid(kernel, grid_side, pad=1):
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


def log_images(name, inputs, outputs=None):
    """Log inputs and outputs batch of images. Display images in grids
    Args:
        name: name of the summary
        inputs: tensor with shape [batch_size, height, widht, depth]
        outputs: if present must have the same dimensions as inputs
    """

    with tf.variable_scope('visualization'):
        batch_size = inputs.get_shape()[0].value
        grid_side = math.floor(math.sqrt(batch_size))
        inputs = on_grid(
            tf.transpose(inputs, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
            grid_side)

        if outputs is None:
            tf_log(tf.summary.image(name, inputs, max_outputs=1))
            return

        inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 10], [0, 0]])
        outputs = on_grid(
            tf.transpose(outputs, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
            grid_side)
        tf_log(
            tf.summary.image(
                name, tf.concat([inputs, outputs], axis=2), max_outputs=1))
