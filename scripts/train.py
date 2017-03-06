#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Dynamically define the train bench via CLI"""

import csv
import os
import sys
import tensorflow as tf

from CLIArgs import CLIArgs
from dtb.train import train


def main():
    """Executes the training procedure and write the results
    to the results.csv file"""
    with tf.device(ARGS.train_device):
        info = train(
            model=MODEL,
            dataset=DATASET,
            hyperparameters={
                "epochs": ARGS.epochs,
                "batch_size": ARGS.batch_size,
                "regularizations": {
                    "l2": ARGS.l2_penalty,
                    "augmentation": lambda image: image,
                },
                "gd": {
                    "optimizer": getattr(tf.train, ARGS.optimizer),
                    "args": ARGS.optimizer_args
                },
                "lr_decay": {
                    "enabled": ARGS.lr_decay,
                    "epochs": ARGS.lr_decay_epochs,
                    "factor": ARGS.lr_decay_factor
                },
            },
            force_restart=ARGS.restart,
            surgery={
                "checkpoint_path": ARGS.checkpoint_path,
                "exclude_scopes": ARGS.exclude_scopes,
                "trainable_scopes": ARGS.trainable_scopes
            },
            comment=ARGS.comment)

    resultsfile = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results.csv')
    writeheader = not os.path.exists(resultsfile)

    with open(resultsfile, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, info.keys())
        if writeheader:
            writer.writeheader()
        writer.writerow(info)
    return 0


if __name__ == '__main__':
    ARGS, MODEL, DATASET = CLIArgs().parse_train()
    sys.exit(main())