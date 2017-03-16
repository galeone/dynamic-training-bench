#!/usr/bin/env python

#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate the model """

import tensorflow as tf
import sys

from dytb.evaluate import evaluate
from dytb.inputs.interfaces import InputType

from CLIArgs import CLIArgs


def main():
    """Evaluates the model, on the specified dataset,
    fetching the requested input type"""
    with tf.device(ARGS.eval_device):
        print(
            evaluate(ARGS.checkpoint_path, MODEL, DATASET, INPUT_TYPE,
                     ARGS.batch_size))


if __name__ == '__main__':
    ARGS, MODEL, DATASET = CLIArgs(
        description="Evaluate the model").parse_eval()

    INPUT_TYPE = InputType.test if ARGS.test else InputType.validation
    sys.exit(main())
