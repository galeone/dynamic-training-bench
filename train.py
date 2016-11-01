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

import argparse
import importlib
import sys
from datetime import datetime
import os.path
import time
import math

import numpy as np
import tensorflow as tf
import evaluate
from models import utils
from decay import supervised_parameter_decay


def train(lr_decay, kp_decay):
    """Train model.
    Args:
        lr_decay: if True decay the learning rate exponentially
        kp_decay: if True use supervised parameter decay to decay keep_prob"""

    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = DATASET.distorted_inputs(BATCH_SIZE)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        keep_prob_, logits = MODEL.get_model(
            images, DATASET.NUM_CLASSES, train_phase=True)

        # Calculate loss.
        loss = MODEL.loss(logits, labels)

        if lr_decay:
            # Decay the learning rate exponentially based on the number of steps.
            learning_rate = tf.train.exponential_decay(
                INITIAL_LEARNING_RATE,
                global_step,
                STEPS_PER_DECAY,
                LEARNING_RATE_DECAY_FACTOR,
                staircase=True)
        else:
            learning_rate = tf.constant(INITIAL_LEARNING_RATE)

        utils.log(tf.scalar_summary('learning_rate', learning_rate))
        train_op = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(
            loss, global_step=global_step)

        # Create the train saver.
        variables_to_save = utils.variables_to_save([global_step])
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
        # va placeholder required for supervised_parameter_decay
        validation_accuracy_ = tf.placeholder(
            tf.float32, shape=(), name="validation_accuracy_")

        if kp_decay:
            # Decay keep prob using supervised parameter decay
            decay_keep_prob = supervised_parameter_decay(
                validation_accuracy_, initial_parameter_value=MAX_KEEP_PROB)
        else:
            decay_keep_prob = tf.constant(MAX_KEEP_PROB)

        keep_prob_summary = tf.scalar_summary('keep_prob', decay_keep_prob)

        # set initial keep_prob
        keep_prob = MAX_KEEP_PROB

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
                        step % STEPS_PER_EPOCH == 0) or (step + 1) == MAX_STEPS:
                    checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
                    train_saver.save(sess, checkpoint_path, global_step=step)

                    # validation accuracy
                    validation_accuracy_value = evaluate.get_validation_accuracy(
                        LOG_DIR, MODEL, DATASET)
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


def method_name(args):
    """Build method name parsing args"""
    name = ''
    if args.kp_decay:
        name += 'kp_decay_'
    if args.lr_decay:
        name += 'exp_lr_'
    name = name.rstrip('_')

    if name == '':
        name = 'constant'
    return name


if __name__ == '__main__':
    MODELS = [
        model[len('models/'):-3] for model in glob.glob('models/*.py')
        if "__init__.py" not in model and "utils" not in model
    ]
    # CLI arguments
    PARSER = argparse.ArgumentParser(description="Train the model")
    PARSER.add_argument("--model", required=True, choices=MODELS)
    PARSER.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100"])
    PARSER.add_argument("--kp_decay", action="store_true")
    PARSER.add_argument("--lr_decay", action="store_true")
    ARGS = PARSER.parse_args()

    # Load required model and dataset
    MODEL = importlib.import_module("models." + ARGS.model)
    DATASET = importlib.import_module("inputs." + ARGS.dataset)

    # Training constants
    BATCH_SIZE = 128
    STEPS_PER_EPOCH = math.ceil(DATASET.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                BATCH_SIZE)
    MAX_EPOCH = 300
    MAX_STEPS = STEPS_PER_EPOCH * MAX_EPOCH
    MOMENTUM = 0.9

    # Learning rate decay constants
    INITIAL_LEARNING_RATE = 1e-2
    NUM_EPOCHS_PER_DECAY = 25  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    STEPS_PER_DECAY = STEPS_PER_EPOCH * NUM_EPOCHS_PER_DECAY

    # Keep prob decay constants
    MAX_KEEP_PROB = 1.0

    # Model logs and checkpoint constants
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(CURRENT_DIR, 'log', ARGS.model, method_name(ARGS))
    BEST_MODEL_DIR = os.path.join(LOG_DIR, 'best')

    # Dataset creation if needed
    DATASET.maybe_download_and_extract()
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    if not tf.gfile.Exists(BEST_MODEL_DIR):
        tf.gfile.MakeDirs(BEST_MODEL_DIR)

    # Start train
    train(ARGS.lr_decay, ARGS.kp_decay)
    sys.exit()
