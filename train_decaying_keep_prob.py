#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
# Based on Tensorflow cifar10_train.py file
# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_train.py
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Train model with a single GPU. Evaluate it on the second one"""

import sys
from datetime import datetime
import os.path
import time
import math

import numpy as np
import tensorflow as tf
from models import model2 as vgg
from inputs import cifar10 as dataset
import evaluate

BATCH_SIZE = 128
STEP_PER_EPOCH = math.ceil(dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                           BATCH_SIZE)
MAX_EPOCH = 300
MAX_STEPS = STEP_PER_EPOCH * MAX_EPOCH

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, 'log', vgg.NAME, 'keep_prob_decay')
BEST_MODEL_DIR = os.path.join(LOG_DIR, 'best')

MAX_KEEP_PROB = 1.0


def keep_prob_decay(validation_accuracy_,
                    max_keep_prob=1.0,
                    min_keep_prob=0.4,
                    num_updates=5,
                    decay_amount=0.05,
                    precision=1e-2,
                    name=None):
    """ Decay keep_prob until it reaches min_keep_pro. Computation
    based on validation_accuracy_ variations.
    """

    with tf.name_scope(name, "KeepProbDecay", [
            validation_accuracy_, max_keep_prob, min_keep_prob, num_updates,
            decay_amount
    ]) as name, tf.device('/cpu:0'):
        # Maintains the state of the computation. Initialized with max_keep_prob
        keep_prob = tf.Variable(
            max_keep_prob, dtype=tf.float32, trainable=False, name="keep_prob")

        num_updates = int(num_updates)
        # crate a tensor with num_updates values, to accumulate validation accuracies
        accumulator = tf.Variable(
            tf.zeros(
                [num_updates], dtype=tf.float32), trainable=False)
        position = tf.Variable(0, dtype=tf.int32, trainable=False)
        accumulated = tf.Variable(0, dtype=tf.int32, trainable=False)

        # keep only the specified precision of validation_accuracy_
        validation_accuracy = tf.Variable(0.0)
        with tf.control_dependencies([
                tf.assign(validation_accuracy,
                          tf.ceil(validation_accuracy_ / precision) * precision)
        ]):
            # trigger value: 0 (nop) or 1 (trigger)
            mean = tf.ceil(
                tf.reduce_sum(accumulator) /
                (num_updates * precision)) * precision
            trigger = 1 - tf.ceil(validation_accuracy - mean)

            # compute next keep prob
            with tf.control_dependencies([mean, trigger]):
                # if trigger, pos = 0, else accumulated % num_updates
                def reset_position():
                    """ reset accumulator vector position """
                    # side effect insided the function
                    with tf.control_dependencies([tf.assign(position, 0)]):
                        return tf.identity(position)

                position = tf.cond(
                    tf.equal(trigger, 1), reset_position,
                    lambda: tf.mod(accumulated, num_updates))

                # execute only after position update
                with tf.control_dependencies([position]):

                    def reset_accumulator():
                        """set past validation accuracies to 0 and place actual
                        validation accuracy in position 0"""
                        with tf.control_dependencies([
                                tf.scatter_update(
                                    accumulator,
                                    [i for i in range(num_updates)],
                                    [validation_accuracy] +
                                    [0.0 for i in range(1, num_updates)])
                        ]):
                            return tf.identity(accumulator)

                    def update_accumulator():
                        """ add the new VA value into the accumulator """
                        with tf.control_dependencies([
                                tf.scatter_update(accumulator, position,
                                                  validation_accuracy)
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

                    updated_keep_prob = tf.assign(
                        keep_prob,
                        tf.maximum(min_keep_prob,
                                   keep_prob - decay_amount * trigger))
                    return updated_keep_prob


def train():
    """Train model"""
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = dataset.distorted_inputs(BATCH_SIZE)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        keep_prob_, logits = vgg.get_model(images, train_phase=True)

        # Calculate loss.
        loss = vgg.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = vgg.train(loss, global_step)

        # Create the train saver.
        variables_to_save = tf.trainable_variables() + [global_step]
        train_saver = tf.train.Saver(variables_to_save)
        # Create the best model saver.
        best_saver = tf.train.Saver(variables_to_save)

        # Train accuracy ops
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        train_accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
        # General validation summary
        accuracy_value_ = tf.placeholder(tf.float32, shape=())
        accuracy_summary = tf.scalar_summary('accuracy', accuracy_value_)

        # Initialize decay_keep_prob op
        # va placeholder required for keep_prob_decay
        validation_accuracy_ = tf.placeholder(
            tf.float32, shape=(), name="validation_accuracy_")
        decay_keep_prob = keep_prob_decay(
            validation_accuracy_, max_keep_prob=MAX_KEEP_PROB)
        keep_prob_summary = tf.scalar_summary('keep_prob', decay_keep_prob)

        # read collection after keep_prob_decay that adds
        # the keep_prob summary
        train_summaries = tf.merge_summary(
            tf.get_collection_ref('train_summaries'))

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            train_log = tf.train.SummaryWriter(LOG_DIR + "/train", sess.graph)
            validation_log = tf.train.SummaryWriter(LOG_DIR + "/validation",
                                                    sess.graph)

            # Extract previous global step value
            old_gs = sess.run(global_step)

            # set initial keep_prob
            keep_prob = MAX_KEEP_PROB
            # set best_validation_accuracy, used by best_saver
            best_validation_accuracy = 0.0

            # Restart from where we were
            for step in range(old_gs, MAX_STEPS):
                start_time = time.time()
                _, loss_value, summary_lines = sess.run(
                    [train_op, loss, train_summaries],
                    feed_dict={keep_prob_: keep_prob})
                duration = time.time() - start_time

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                # update logs every 10 iterations
                if step % 10 == 0:
                    num_examples_per_step = BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('{}: step {}, loss = {:.2f} '
                                  '({:.1f} examples/sec; {:.3f} sec/batch)')
                    print(
                        format_str.format(datetime.now(), step, loss_value,
                                          examples_per_sec, sec_per_batch))
                    # log train values
                    train_log.add_summary(summary_lines, global_step=step)

                # Save the model checkpoint at the end of every epoch
                # evaluate train and validation performance
                if (step > 0 and
                        step % STEP_PER_EPOCH == 0) or (step + 1) == MAX_STEPS:
                    checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
                    train_saver.save(sess, checkpoint_path, global_step=step)

                    # validation accuracy
                    validation_accuracy_value = evaluate.get_validation_accuracy(
                        LOG_DIR)
                    summary_line = sess.run(
                        accuracy_summary,
                        feed_dict={accuracy_value_: validation_accuracy_value})
                    validation_log.add_summary(summary_line, global_step=step)

                    # update keep_prob using new validation accuracy
                    keep_prob, summary_line = sess.run(
                        [decay_keep_prob, keep_prob_summary],
                        feed_dict={
                            validation_accuracy_: validation_accuracy_value
                        })
                    train_log.add_summary(summary_line, global_step=step)

                    # train accuracy
                    train_accuracy_value = sess.run(
                        train_accuracy, feed_dict={keep_prob_: 1.0})
                    summary_line = sess.run(
                        accuracy_summary,
                        feed_dict={accuracy_value_: train_accuracy_value})
                    train_log.add_summary(summary_line, global_step=step)

                    print(('{}: train accuracy = {:.3f}\n'
                           'validation accuracy = {:.3f}\n'
                           'keep_prob = {:.2f}').format(datetime.now(
                           ), train_accuracy_value, validation_accuracy_value,
                                                        keep_prob))
                    # save best model
                    if validation_accuracy_value > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy_value
                        # fixed global_step, the best model is only one
                        best_saver.save(
                            sess,
                            os.path.join(BEST_MODEL_DIR, 'model.ckpt'),
                            global_step=0)


def main():
    """main function"""
    dataset.maybe_download_and_extract()
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    if not tf.gfile.Exists(BEST_MODEL_DIR):
        tf.gfile.MakeDirs(BEST_MODEL_DIR)
    train()
    return 0


if __name__ == '__main__':
    sys.exit(main())
