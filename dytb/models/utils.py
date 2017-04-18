#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utils for models creation"""

import re
import tensorflow as tf
from .collections import MODEL_SUMMARIES, REQUIRED_NON_TRAINABLES


def legalize_name(name):
    """Made name a legal name to be used in tensorflow summaries
    Args:
        name: string
    Returns:
        name_legal
    """
    return re.sub(r"[^\w|/]", "_", name)


def tf_log(summary, collection=MODEL_SUMMARIES):
    """Add tf.summary object to collection named collection"""
    tf.add_to_collection(collection, summary)


def training_process_variables():
    """global variables - trainable variables:
    it contains the variable defined by the optimizer and the others
    defined durint the training process.
    Those variables are useful when restoring a training process.
    Those variables are not trainable. can be saved and restored.
    """

    return [
        variable for variable in tf.global_variables()
        if variable.name not in (var.name for var in tf.trainable_variables())
    ]


def variables_to_save(add_list=None):
    """Returns a list of variables to save.
    add_list variables are always added to the list
    Args:
        add_list: a list of variables
    Returns:
        list: list of tensors to save
    """
    if add_list is None:
        add_list = []
    return tf.trainable_variables() + tf.get_collection_ref(
        REQUIRED_NON_TRAINABLES) + add_list + training_process_variables()


def variables_to_restore(add_list=None, exclude_scope_list=None):
    """Returns a list of variables to restore to made the model working
    properly.
    The list is made by the trainable variables + required non trainable variables
    such as statistics of batch norm layers.
    Remove from the list variables that are in the exclude_scope_list.
    Add variables in the add_list

    Args:
        add_list: a list of variables
        exclude_scope_list: a list of scopes to exclude
    Returns:
        list: list of tensors to restore
    """

    if add_list is None:
        add_list = []
    if exclude_scope_list is None:
        exclude_scope_list = []

    variables = variables_to_save()
    if exclude_scope_list:
        variables[:] = [
            variable for variable in variables
            if not variable.name.startswith(
                tuple(scope for scope in exclude_scope_list))
        ]
    return variables + add_list


def variables_to_train(scope_list=None):
    """Returns a list of variables to train, filtered by the scopes.
    Args:
        scope_list: a list of scope to train
    Returns:
        the list of variables to train by the optimizer
    """
    if scope_list is None:
        return tf.trainable_variables()
    vars_to_train = []
    for scope in scope_list:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        vars_to_train.extend(variables)
    return vars_to_train


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


def count_trainable_parameters(print_model=False):
    """Count the number of trainable parameters is the current graph.
    Returns:
        count: the number of trainable parameters"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        if print_model:
            print(variable)
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters
