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
import json
import glob
import importlib
import pprint
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


def train():
    """Train model.

    Returns:
        best validation accuracy obtained. Save best model"""

    best_validation_accuracy = 0.0

    with tf.Graph().as_default(), tf.device(DEVICE):
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = DATASET.distorted_inputs(BATCH_SIZE)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        keep_prob_, logits = MODEL.get_model(
            images,
            DATASET.NUM_CLASSES,
            train_phase=True,
            l2_penalty=L2_PENALTY)

        # Calculate loss.
        loss = MODEL.loss(logits, labels)

        if LR_DECAY:
            # Decay the learning rate exponentially based on the number of steps.
            learning_rate = tf.train.exponential_decay(
                INITIAL_LR,
                global_step,
                STEPS_PER_DECAY,
                LR_DECAY_FACTOR,
                staircase=True)
        else:
            learning_rate = tf.constant(INITIAL_LR)

        utils.log(tf.scalar_summary('learning_rate', learning_rate))
        train_op = OPTIMIZER.minimize(loss, global_step=global_step)

        # Create the train saver.
        variables_to_save = utils.variables_to_save([global_step])
        train_saver = tf.train.Saver(variables_to_save, max_to_keep=2)
        # Create the best model saver.
        best_saver = tf.train.Saver(variables_to_save, max_to_keep=1)

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

        if KP_DECAY:
            # Decay keep prob using supervised parameter decay
            decay_keep_prob = supervised_parameter_decay(
                validation_accuracy_,
                initial_parameter_value=INITIAL_KP,
                min_parameter_value=FINAL_KP,
                num_observations=NUM_OBSERVATIONS,
                decay_amount=KP_DECAY_AMOUNT)
            # set initial keep prob
            keep_prob = INITIAL_KP
        else:
            # if kp is not decayed, a useless value is passed to the
            # keep_prob placeholder.
            # Ops that uses this placeholder will use this value
            # Whilst Ops that have hardcoded keep_prob value, will use
            # those.
            decay_keep_prob = tf.constant(1.0)
            keep_prob = 1.0

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
                        best_saver.save(
                            sess,
                            os.path.join(BEST_MODEL_DIR, 'model.ckpt'),
                            global_step=step)
    return best_validation_accuracy


def method_name():
    """Build method name parsing args"""
    name = '{}_{}_lr={:.6f}_'.format(ARGS.dataset, OPTIMIZER._name, INITIAL_LR)
    if LR_DECAY:
        name += 'exp_lr_'
    if KP_DECAY:
        name += 'kp_decay_'
    if L2_PENALTY != 0.0:
        name += 'l2={:.6f}_'.format(L2_PENALTY)

    return name.rstrip('_')


if __name__ == '__main__':
    MODELS = [
        model[len('models/'):-3] for model in glob.glob('models/*.py')
        if "__init__.py" not in model and "utils" not in model
    ]

    DATASETS = [
        dataset[len('inputs/'):-3] for dataset in glob.glob('inputs/*.py')
        if "__init__.py" not in dataset
    ]

    OPTIMIZERS = [
        optimizer for optimizer in dir(tf.train)
        if optimizer.endswith("Optimizer")
    ]

    # CLI arguments
    PARSER = argparse.ArgumentParser(description="Train the model")

    # Required arguments
    PARSER.add_argument("--model", required=True, choices=MODELS)
    PARSER.add_argument("--dataset", required=True, choices=DATASETS)

    # Learning rate decay arguments
    PARSER.add_argument("--lr_decay", action="store_true")
    PARSER.add_argument("--lr_decay_epochs", type=int, default=25)
    PARSER.add_argument("--lr_decay_factor", type=float, default=0.1)

    # Keep prob decay arguments
    PARSER.add_argument("--kp_decay", action="store_true")
    PARSER.add_argument("--initial_kp", type=float, default=1.0)
    PARSER.add_argument("--final_kp", type=float, default=0.5)
    PARSER.add_argument("--num_observations", type=int, default=25)
    PARSER.add_argument("--kp_decay_amount", type=float, default=0.5)

    # L2 regularization arguments
    PARSER.add_argument("--l2_penalty", type=float, default=0.0)

    # Optimization arguments
    PARSER.add_argument(
        "--optimizer", choices=OPTIMIZERS, default="MomentumOptimizer")
    PARSER.add_argument(
        "--optimizer_args",
        type=json.loads,
        default='''
    {
        "learning_rate": 1e-2,
        "momentum": 0.9
    }''')
    PARSER.add_argument("--batch_size", type=int, default=128)
    PARSER.add_argument("--epochs", type=int, default=300)

    # Hardware
    PARSER.add_argument("--device", default="/gpu:0")

    # Pargse arguments
    ARGS = PARSER.parse_args()

    # Load required model and dataset
    MODEL = importlib.import_module("models." + ARGS.model)
    DATASET = importlib.import_module("inputs." + ARGS.dataset)

    # Training constants
    OPTIMIZER = getattr(tf.train, ARGS.optimizer)(**ARGS.optimizer_args)
    # Learning rate must be always present in optimizer args
    INITIAL_LR = float(ARGS.optimizer_args["learning_rate"])

    BATCH_SIZE = ARGS.batch_size
    MAX_EPOCH = ARGS.epochs
    STEPS_PER_EPOCH = math.ceil(DATASET.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                BATCH_SIZE)
    MAX_STEPS = STEPS_PER_EPOCH * MAX_EPOCH

    # Regularization constaints
    L2_PENALTY = ARGS.l2_penalty

    # Learning rate decay constants
    LR_DECAY = False
    if ARGS.lr_decay:
        LR_DECAY = True
        NUM_EPOCHS_PER_DECAY = ARGS.lr_decay_epochs
        LR_DECAY_FACTOR = ARGS.lr_decay_factor
        STEPS_PER_DECAY = STEPS_PER_EPOCH * NUM_EPOCHS_PER_DECAY

    # Keep prob decay constants
    KP_DECAY = False
    if ARGS.kp_decay:
        KP_DECAY = True
        INITIAL_KP = ARGS.initial_kp
        FINAL_KP = ARGS.final_kp
        NUM_OBSERVATIONS = ARGS.num_observations
        KP_DECAY_AMOUNT = ARGS.kp_decay_amount

    # Model logs and checkpoint constants
    NAME = method_name()
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(CURRENT_DIR, 'log', ARGS.model, NAME)
    BEST_MODEL_DIR = os.path.join(LOG_DIR, 'best')

    # Device where to place the model
    DEVICE = ARGS.device

    # Dataset creation if needed
    DATASET.maybe_download_and_extract()
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    if not tf.gfile.Exists(BEST_MODEL_DIR):
        tf.gfile.MakeDirs(BEST_MODEL_DIR)

    # Start train, get best validation accuracy at the end
    pprint.pprint(ARGS)
    BEST_VA = train()
    with open(os.path.join(CURRENT_DIR, "results.txt"), "a") as res:
        res.write("{}: {} {}\n".format(ARGS.model, NAME, BEST_VA))
    sys.exit()
