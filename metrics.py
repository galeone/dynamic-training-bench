#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utility functions for model training and evaluation"""

import tensorflow as tf


def accuracy_op(logits, labels):
    """Define the accuracy between predictions (logits) and labels.
    Args:
        logits: a [batch_size, 1,1, num_classes] tensor or
                a [batch_size, num_classes] tensor
        labels: a [batch_size] tensor
    Returns:
        accuracy: the accuracy op
    """

    with tf.variable_scope('accuracy'):
        # handle fully convolutional classifiers
        logits_shape = logits.shape
        if len(logits_shape) == 4 and logits_shape[1:3] == [1, 1]:
            top_k_logits = tf.squeeze(logits, [1, 2])
        else:
            top_k_logits = logits
        top_k_op = tf.nn.in_top_k(top_k_logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

    return accuracy


def iou_op(real_coordinates, coordinates):
    """Returns the average interserction over union operation between a batch of
    real_coordinates and a batch of coordinates.
    Args:
        real_coordinates: a tensor with shape [batch_size, 4]
        coordinates: a tensor with shape [batch_size, 4]
    Returns:
        iou: avewrage interserction over union in the batch
    """

    with tf.variable_scope('iou'):
        ymin_orig = real_coordinates[:, 0]
        xmin_orig = real_coordinates[:, 1]
        ymax_orig = real_coordinates[:, 2]
        xmax_orig = real_coordinates[:, 3]
        area_orig = (ymax_orig - ymin_orig) * (xmax_orig - xmin_orig)

        ymin = coordinates[:, 0]
        xmin = coordinates[:, 1]
        ymax = coordinates[:, 2]
        xmax = coordinates[:, 3]
        area_pred = (ymax - ymin) * (xmax - xmin)

        intersection_ymin = tf.maximum(ymin, ymin_orig)
        intersection_xmin = tf.maximum(xmin, xmin_orig)
        intersection_ymax = tf.minimum(ymax, ymax_orig)
        intersection_xmax = tf.minimum(xmax, xmax_orig)

        intersection_area = tf.maximum(
            intersection_ymax - intersection_ymin,
            tf.zeros_like(intersection_ymax)) * tf.maximum(
                intersection_xmax - intersection_xmin,
                tf.zeros_like(intersection_ymax))

        iou = tf.reduce_mean(intersection_area /
                             (area_orig + area_pred - intersection_area))
        return iou
