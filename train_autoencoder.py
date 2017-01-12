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
import evaluate
from models.utils import variables_to_save, tf_log, MODEL_SUMMARIES
from models.utils import put_kernels_on_grid
from inputs.utils import InputType
from CLIArgs import CLIArgs


def train():
    """Train model.

    Returns:
        best validation error. Save best model"""

    best_validation_error_value = float('inf')

    with tf.Graph().as_default(), tf.device(ARGS.train_device):
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and discard labels
        images, _ = DATASET.distorted_inputs(ARGS.batch_size)

        # Build a Graph that computes the reconstructions predictions from the
        # inference model.
        is_training_, reconstructions = MODEL.get(images,
                                                  train_phase=True,
                                                  l2_penalty=ARGS.l2_penalty)

        # display original images next to reconstructed images
        with tf.variable_scope('visualization'):
            grid_side = math.floor(math.sqrt(ARGS.batch_size))
            inputs = put_kernels_on_grid(
                tf.transpose(
                    images, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
                grid_side)
            inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 10], [0, 0]])
            outputs = put_kernels_on_grid(
                tf.transpose(
                    reconstructions,
                    perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
                grid_side)
        tf_log(
            tf.summary.image(
                'input_output', tf.concat(2, [inputs, outputs]), max_outputs=1))

        # Calculate loss.
        loss = MODEL.loss(reconstructions, images)
        # reconstruction error
        error_ = tf.placeholder(tf.float32, shape=())
        error = tf.summary.scalar('error', error_)

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
        # Create the best model saver
        best_saver = tf.train.Saver(variables, max_to_keep=1)

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
                os.path.join(LOG_DIR, str(InputType.train)), graph=sess.graph)
            validation_log = tf.summary.FileWriter(
                os.path.join(LOG_DIR, str(InputType.validation)),
                graph=sess.graph)

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
                    num_examples_per_step = ARGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('{}: step {}, loss = {:.4f} '
                                  '({:.1f} examples/sec; {:.3f} sec/batch)')
                    print(
                        format_str.format(datetime.now(), step, loss_value,
                                          examples_per_sec, sec_per_batch))
                    # log train error and summaries
                    train_error_summary_line, train_summary_line = sess.run(
                        [error, train_summaries],
                        feed_dict={error_: loss_value,
                                   is_training_: True})
                    train_log.add_summary(
                        train_error_summary_line, global_step=step)
                    train_log.add_summary(train_summary_line, global_step=step)

                # Save the model checkpoint at the end of every epoch
                # evaluate train and validation performance
                if (step > 0 and
                        step % STEPS_PER_EPOCH == 0) or (step + 1) == MAX_STEPS:
                    checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
                    train_saver.save(sess, checkpoint_path, global_step=step)

                    # validation error
                    validation_error_value = evaluate.error(
                        LOG_DIR,
                        MODEL,
                        DATASET,
                        InputType.validation,
                        device=ARGS.eval_device)

                    summary_line = sess.run(
                        error, feed_dict={error_: validation_error_value})
                    validation_log.add_summary(summary_line, global_step=step)

                    print('{} ({}): train error = {} validation error = {}'.
                          format(datetime.now(),
                                 int(step / STEPS_PER_EPOCH), loss_value,
                                 validation_error_value))
                    if validation_error_value < best_validation_error_value:
                        best_validation_error_value = validation_error_value
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
    return best_validation_error_value


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

    # Start train and get best (lower) error of the training process
    BEST_ERROR = train()

    # Save the best error value on the validation set
    with open(os.path.join(CURRENT_DIR, 'validation_results.txt'), 'a') as res:
        res.write('{} {}: {} {}\n'.format(datetime.now(), ARGS.model, NAME,
                                          BEST_ERROR))

    # Use the 'best' model to calculat the error on the test set
    with open(os.path.join(CURRENT_DIR, 'test_results.txt'), 'a') as res:
        res.write('{} {}: {} {}\n'.format(
            datetime.now(),
            ARGS.model,
            NAME,
            evaluate.error(
                BEST_MODEL_DIR,
                MODEL,
                DATASET,
                InputType.test,
                device=ARGS.eval_device)))
    sys.exit()
