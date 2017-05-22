#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Evaluation method and utilities"""

from .inputs.interfaces import InputType


def evaluate(metric,
             checkpoint_path,
             model,
             dataset,
             input_type,
             batch_size,
             augmentation_fn=None):
    """Eval the model, restoring weight found in checkpoint_path, using the dataset.
    Args:
        metric: the metric to evaluate. The usual dictionary with the fn and its properties
        checkpoint_path: path of the trained model checkpoint directory
        model: implementation of the Model interface
        dataset: implementation of the Input interface
        input_type: InputType enum
        batch_size: evaluate in batch of size batch_size
        augmentation_fn: if present, applies the augmentation to the input data
    Returns:
        value: scalar value representing the evaluation of the model,
               on the dataset, fetching values of the specified input_type
    """
    InputType.check(input_type)
    model.evaluator.dataset = dataset
    return model.evaluator.eval(metric, checkpoint_path, input_type, batch_size,
                                augmentation_fn)


def stats(checkpoint_path, model, dataset, batch_size, augmentation_fn=None):
    """Eval the model, restoring weight found in checkpoint_path, using the dataset.
    Args:
        checkpoint_path: path of the trained model checkpoint directory
        model: implementation of the Model interface
        dataset: implementation of the Input interface
        batch_size: evaluate in batch of size batch_size
        augmentation_fn: if present, applies the augmentation to the input data
    Returns:
        dict: a dictionary with the statistics measured
    """
    model.evaluator.dataset = dataset
    return model.evaluator.stats(checkpoint_path, batch_size, augmentation_fn)
