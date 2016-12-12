#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
# Based on Tensorflow cifar10_train.py file
# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_train.py
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate the model """

import os
import argparse
import importlib
from datetime import datetime
import math

import numpy as np
import tensorflow as tf
from inputs.utils import InputType
from models.utils import MODEL_SUMMARIES, tf_log, put_kernels_on_grid
import utils


def accuracy(checkpoint_dir, model, dataset, input_type, device="/gpu:0"):
    """
    Read latest saved checkpoint and use it to evaluate the model
    Args:
        checkpoint_dir: checkpoint folder
        model: python package containing the model to save
        dataset: python package containing the dataset to use
        input_type: InputType enum, the input type of the input examples
        device: device where to place the model and run the evaluation
    """
    if not isinstance(input_type, InputType):
        raise ValueError("Invalid input_type, required a valid type")

    with tf.Graph().as_default(), tf.device(device):
        # Get images and labels from the dataset
        # Use batch_size multiple of train set size and big enough to stay in GPU
        batch_size = 200
        images, labels = dataset.inputs(
            input_type=input_type, batch_size=batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        _, logits = model.get(images, dataset.num_classes(), train_phase=False)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver()
        accuracy_value = 0.0
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('[!] No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for queue_runner in tf.get_collection(
                        tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(
                        queue_runner.create_threads(
                            sess, coord=coord, daemon=True, start=True))

                num_iter = int(
                    math.ceil(dataset.num_examples(input_type) / batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run(top_k_op)
                    true_count += np.sum(predictions)
                    step += 1

                accuracy_value = true_count / total_sample_count
            except Exception as exc:
                coord.request_stop(exc)
            finally:
                coord.request_stop()

            coord.join(threads)
        return accuracy_value


if __name__ == '__main__':
    # CLI arguments
    PARSER = argparse.ArgumentParser(description="Evaluate the model")

    # Required arguments
    PARSER.add_argument("--model", required=True, choices=utils.get_models())
    PARSER.add_argument(
        "--dataset", required=True, choices=utils.get_datasets())
    PARSER.add_argument("--checkpoint_dir", required=True)
    PARSER.add_argument("--test", action="store_true")
    PARSER.add_argument("--device", default="/gpu:0")
    ARGS = PARSER.parse_args()

    # Load required model and dataset, ovverides default
    MODEL = getattr(
        importlib.import_module("models." + ARGS.model), ARGS.model)()
    DATASET = getattr(
        importlib.import_module("inputs." + ARGS.dataset), ARGS.dataset)()

    DATASET.maybe_download_and_extract()
    print('{}: {} accuracy = {:.3f}'.format(
        datetime.now(),
        'test' if ARGS.test else 'validation',
        accuracy(
            ARGS.checkpoint_dir,
            MODEL,
            DATASET,
            InputType.test if ARGS.test else InputType.validation,
            device=ARGS.device)))
