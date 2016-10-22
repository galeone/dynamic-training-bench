# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or iamplied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation for CIFAR-10."""

import sys
from datetime import datetime
import math
import numpy as np
import tensorflow as tf
from inputs import cifar10
import train
import vgg

EVAL_DIR = train.CURRENT_DIR + '/eval'


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt = tf.train.get_checkpoint_state(train.LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[
                -1]
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
                          train.BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * train.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""

    with tf.Graph().as_default() as graph, tf.device('/gpu:1'):
        # Get images and labels for CIFAR-10.
        # Use batch_size multiple of train set size and big enough to stay in GPU
        images, labels = cifar10.inputs(eval_data=True, batch_size=train.BATCH_SIZE)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        _, logits = vgg.get_model(images, train_phase=False)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # saver
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(EVAL_DIR, graph)
        eval_once(saver, summary_writer, top_k_op, summary_op)


def main():
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
    tf.gfile.MakeDirs(EVAL_DIR)
    evaluate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
