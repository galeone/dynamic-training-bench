#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
# Based on Tensorflow cifar10_train.py file
# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_train.py
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Contains the decay formula implementation"""

import tensorflow as tf


def supervised_parameter_decay(metric_value_,
                               initial_parameter_value=1.0,
                               min_parameter_value=0.4,
                               num_observations=10,
                               decay_amount=0.05,
                               precision=1e-3,
                               name=None):
    """ Decay parameter until it reaches min_keep_pro. Computation
    based on metric_value_ variations.
    """

    with tf.name_scope(name, "SupervisedParameterDecay", [
            metric_value_, initial_parameter_value, min_parameter_value,
            num_observations, decay_amount
    ]) as name, tf.device('/cpu:0'):
        # Maintains the state of the computation. Initialized with initial_parameter_value
        parameter = tf.Variable(
            initial_parameter_value,
            dtype=tf.float32,
            trainable=False,
            name="parameter")

        num_observations = int(num_observations)
        # crate a tensor with num_observations values, to accumulate validation accuracies
        accumulator = tf.Variable(
            tf.zeros(
                [num_observations], dtype=tf.float32), trainable=False)
        position = tf.Variable(0, dtype=tf.int32, trainable=False)
        accumulated = tf.Variable(0, dtype=tf.int32, trainable=False)

        # keep only the specified precision of metric_value_
        metric_value = tf.Variable(0.0)
        with tf.control_dependencies([
                tf.assign(metric_value,
                          tf.ceil(metric_value_ / precision) * precision)
        ]):
            # trigger value: 0 (nop) or 1 (trigger)
            mean = tf.ceil(
                tf.reduce_sum(accumulator) /
                (num_observations * precision)) * precision
            trigger = 1 - tf.ceil(metric_value - mean)

            # compute next keep prob
            with tf.control_dependencies([mean, trigger]):
                # if trigger, pos = 0, else accumulated % num_observations
                def reset_position():
                    """ reset accumulator vector position """
                    # side effect insided the function
                    with tf.control_dependencies([tf.assign(position, 0)]):
                        return tf.identity(position)

                position = tf.cond(
                    tf.equal(trigger, 1), reset_position,
                    lambda: tf.mod(accumulated, num_observations))

                # execute only after position update
                with tf.control_dependencies([position]):

                    def reset_accumulator():
                        """set past validation accuracies to 0 and place actual
                        validation accuracy in position 0"""
                        with tf.control_dependencies([
                                tf.scatter_update(
                                    accumulator,
                                    [i for i in range(num_observations)],
                                    [metric_value] +
                                    [0.0 for i in range(1, num_observations)])
                        ]):
                            return tf.identity(accumulator)

                    def update_accumulator():
                        """ add the new VA value into the accumulator """
                        with tf.control_dependencies([
                                tf.scatter_update(accumulator, position,
                                                  metric_value)
                        ]):
                            return tf.identity(accumulator)

                    # update accumulator
                    # if trigger: reset_acculator, else accumulator[position] = va
                    accumulator = tf.cond(
                        tf.equal(trigger, 1), reset_accumulator,
                        update_accumulator)

                    def reset_accumulated():
                        """ reset accumulated counter """
                        with tf.control_dependencies(
                            [tf.assign(accumulated, 1)]):
                            return tf.identity(accumulated)

                    def update_accumulated():
                        """ add 1 to accumulated counter """
                        with tf.control_dependencies(
                            [tf.assign_add(accumulated, 1)]):
                            return tf.identity(accumulated)

                    # update accumulated (for current prob)
                    # if trigger; accumulated = 1, else accumulated +=1
                    accumulated = tf.cond(
                        tf.equal(trigger, 1), reset_accumulated,
                        update_accumulated)

                    with tf.control_dependencies([accumulator, accumulated]):
                        updated_parameter = tf.assign(
                            parameter,
                            tf.maximum(min_parameter_value,
                                       parameter - decay_amount * trigger))
                        return updated_parameter
