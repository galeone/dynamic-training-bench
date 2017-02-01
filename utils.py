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
