#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
# Based on Tensorflow cifar10_eval.py file
# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_eval.py
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Evaluate the model"""

import sys
from datetime import datetime
import math
import numpy as np
import tensorflow as tf
from inputs import cifar10
import train
# evaluate current training model
# therefore we can use every model
# because the structure of each model, differs just in
# the training phase. We use train.LOG_DIR because it points
# to the current training model logs (and checkpoints)
# like ./log/model<N>
from models import model1 as vgg

BATCH_SIZE = 50


def get_acuracy(top_k_op):
    """Run Eval once.

    Args:
      top_k_op: Top K op.
    """
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt = tf.train.get_checkpoint_state(train.LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(
                    qr.create_threads(
                        sess, coord=coord, daemon=True, start=True))

            num_iter = int(
                math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL /
                          BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            accuracy = true_count / total_sample_count
            print('%s: accuracy @ 1 = %.3f' % (datetime.now(), accuracy))
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()

        coord.join(threads)


def evaluate():
    """Eval the model"""

    with tf.Graph().as_default(), tf.device('/gpu:1'):
        # Get images and labels for CIFAR-10.
        # Use batch_size multiple of train set size and big enough to stay in GPU
        images, labels = cifar10.inputs(eval_data=True, batch_size=BATCH_SIZE)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        _, logits = vgg.get_model(images, train_phase=False)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        get_acuracy(top_k_op)


def main():
    """ main function """
    cifar10.maybe_download_and_extract()
    evaluate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
