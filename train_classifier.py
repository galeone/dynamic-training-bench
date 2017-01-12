#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Dynamically define the train bench via CLI. Specify the dataset to use, the model to train
and any other hyper-parameter"""

import sys
from datetime import datetime
import os.path
import time
import math

import numpy as np
import tensorflow as tf
import evaluate_classifier as evaluate
from models.utils import variables_to_save, tf_log, MODEL_SUMMARIES
from models.utils import put_kernels_on_grid
from inputs.utils import InputType
from CLIArgs import CLIArgs


def train():
    """Train model.

    Returns:
        best validation accuracy obtained. Save best model"""

    best_validation_accuracy = 0.0

    with tf.Graph().as_default(), tf.device(ARGS.train_device):
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels
        images, labels = DATASET.distorted_inputs(ARGS.batch_size)

        with tf.variable_scope('visualization'):
            grid_side = math.floor(math.sqrt(ARGS.batch_size))
            inputs = put_kernels_on_grid(
                tf.transpose(
                    images, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
                grid_side)

        tf_log(tf.summary.image('inputs', inputs, max_outputs=1))

        # Build a Graph that computes the logits predictions from the
        # inference model.
        is_training_, logits = MODEL.get(images,
                                         DATASET.num_classes(),
                                         train_phase=True,
                                         l2_penalty=ARGS.l2_penalty)

        # Calculate loss.
        loss = MODEL.loss(logits, labels)
        tf_log(tf.summary.scalar('loss', loss))

        # learning rate
        initial_lr = float(ARGS.optimizer_args['learning_rate'])

        if ARGS.lr_decay:
            # Decay the learning rate exponentially based on the number of steps.
            steps_per_decay = STEPS_PER_EPOCH * ARGS.lr_decay_epochs
            learning_rate = tf.train.exponential_decay(
                initial_lr,
                global_step,
                steps_per_decay,
                ARGS.lr_decay_factor,
                staircase=True)
        else:
            learning_rate = tf.constant(initial_lr)

        tf_log(tf.summary.scalar('learning_rate', learning_rate))
        train_op = OPTIMIZER.minimize(loss, global_step=global_step)

        # Create the train saver.
        variables = variables_to_save([global_step])
        train_saver = tf.train.Saver(variables, max_to_keep=2)
        # Create the best model saver.
        best_saver = tf.train.Saver(variables, max_to_keep=1)

        # Train accuracy ops
        with tf.variable_scope('accuracy'):
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            train_accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
            # General validation summary
            accuracy_value_ = tf.placeholder(tf.float32, shape=())
            accuracy_summary = tf.summary.scalar('accuracy', accuracy_value_)

        # read collection after that every op added its own
        # summaries in the train_summaries collection
        train_summaries = tf.summary.merge(
            tf.get_collection_ref(MODEL_SUMMARIES))

        # Build an initialization operation to run below.
        init = tf.variables_initializer(tf.global_variables() +
                                        tf.local_variables())

        # Start running operations on the Graph.
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(init)

            # Start the queue runners with a coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if not ARGS.restart:  # continue from the saved checkpoint
                # restore previous session if exists
                checkpoint = tf.train.latest_checkpoint(LOG_DIR)
                if checkpoint:
                    train_saver.restore(sess, checkpoint)
                else:
                    print('[I] Unable to restore from checkpoint')

            train_log = tf.summary.FileWriter(
                os.path.join(LOG_DIR, 'train'), graph=sess.graph)
            validation_log = tf.summary.FileWriter(
                os.path.join(LOG_DIR, 'validation'), graph=sess.graph)

            # Extract previous global step value
            old_gs = sess.run(global_step)

            # Restart from where we were
            for step in range(old_gs, MAX_STEPS):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={is_training_: True})
                duration = time.time() - start_time

                if np.isnan(loss_value):
                    print('Model diverged with loss = NaN')
                    break

                # update logs every 10 iterations
                if step % 10 == 0:
                    examples_per_sec = ARGS.batch_size / duration
                    sec_per_batch = float(duration)

                    format_str = ('{}: step {}, loss = {:.4f} '
                                  '({:.1f} examples/sec; {:.3f} sec/batch)')
                    print(
                        format_str.format(datetime.now(), step, loss_value,
                                          examples_per_sec, sec_per_batch))
                    # log train values
                    summary_lines = sess.run(train_summaries,
                                             feed_dict={is_training_: True})
                    train_log.add_summary(summary_lines, global_step=step)

                # Save the model checkpoint at the end of every epoch
                # evaluate train and validation performance
                if (step > 0 and
                        step % STEPS_PER_EPOCH == 0) or (step + 1) == MAX_STEPS:
                    checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
                    train_saver.save(sess, checkpoint_path, global_step=step)

                    # validation accuracy
                    validation_accuracy_value = evaluate.accuracy(
                        LOG_DIR,
                        MODEL,
                        DATASET,
                        InputType.validation,
                        device=ARGS.eval_device)

                    summary_line = sess.run(
                        accuracy_summary,
                        feed_dict={accuracy_value_: validation_accuracy_value})
                    validation_log.add_summary(summary_line, global_step=step)

                    # train accuracy
                    train_accuracy_value = sess.run(
                        train_accuracy, feed_dict={is_training_: False})
                    summary_line = sess.run(
                        accuracy_summary,
                        feed_dict={accuracy_value_: train_accuracy_value})
                    train_log.add_summary(summary_line, global_step=step)

                    print(
                        '{} ({}): train accuracy = {:.3f} validation accuracy = {:.3f}'.
                        format(datetime.now(),
                               int(step / STEPS_PER_EPOCH),
                               train_accuracy_value, validation_accuracy_value))
                    # save best model
                    if validation_accuracy_value > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy_value
                        best_saver.save(
                            sess,
                            os.path.join(BEST_MODEL_DIR, 'model.ckpt'),
                            global_step=step)
            # end of for

            validation_log.close()
            train_log.close()

            # When done, ask the threads to stop.
            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)
    return best_validation_accuracy


if __name__ == '__main__':
    ARGS, NAME, MODEL, DATASET, OPTIMIZER = CLIArgs().parse_train()

    #### Training constants ####
    STEPS_PER_EPOCH = math.ceil(
        DATASET.num_examples(InputType.train) / ARGS.batch_size)
    MAX_STEPS = STEPS_PER_EPOCH * ARGS.epochs

    #### Model logs and checkpoint constants ####
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(CURRENT_DIR, 'log', ARGS.model, NAME)
    BEST_MODEL_DIR = os.path.join(LOG_DIR, 'best')

    #### Dataset and logs ####
    DATASET.maybe_download_and_extract()

    if tf.gfile.Exists(LOG_DIR) and ARGS.restart:
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    if not tf.gfile.Exists(BEST_MODEL_DIR):
        tf.gfile.MakeDirs(BEST_MODEL_DIR)

    # Train and get best validation accuracy of the training process
    BEST_VA = train()
    with open(os.path.join(CURRENT_DIR, 'validation_results.txt'), 'a') as res:
        res.write('{} {}: {} {}\n'.format(datetime.now(), ARGS.model, NAME,
                                          BEST_VA))

    # Save the test result of the model with the best valiation accuracy
    # the 'best' model
    with open(os.path.join(CURRENT_DIR, 'test_results.txt'), 'a') as res:
        res.write('{} {}: {} {}\n'.format(
            datetime.now(),
            ARGS.model,
            NAME,
            evaluate.accuracy(
                BEST_MODEL_DIR,
                MODEL,
                DATASET,
                InputType.test,
                device=ARGS.eval_device)))

    sys.exit()
