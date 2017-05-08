#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Trainer for the  model"""

import time
import os
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
from .utils import builders, flow

from ..inputs.interfaces import InputType
from ..models.utils import tf_log, variables_to_train, count_trainable_parameters
from ..models.collections import MODEL_SUMMARIES
from ..models.visualization import log_io


class Trainer(object):
    """Trainer for a custom model"""

    def __init__(self, model, dataset, args, steps, paths):
        """Initialize the trainer.
        Args:
            model: the model to train
            dataset: implementation of the Input interface
            args: dictionary of hyperparameters a train parameters
            steps: dictionary of the training steps
            paths: dictionary of the paths
        """
        self._model = model
        self._dataset = dataset
        self._args = args
        self._steps = steps
        self._paths = paths

    def train(self):
        """Train the model
        Returns:
            info: dict containing the information of the trained model
        Side effect:
            saves the latest checkpoints and the best model in its own folder
        """

        with tf.Graph().as_default():
            if self._args["seed"] is not None:
                tf.set_random_seed(self._args["seed"])
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Get inputs and targets: inputs is an input batch
            # target could be either an array of elements or a tensor.
            # it could be [label] or [label, attr1, attr2, ...]
            # or Tensor, where tensor is a standard tensorflow Tensor with
            # its own shape
            return_type = list
            with tf.device('/cpu:0'):
                inputs, targets = self._dataset.inputs(
                    input_type=InputType.train,
                    batch_size=self._args["batch_size"],
                    augmentation_fn=self._args["regularizations"][
                        "augmentation"]["fn"])
                if isinstance(targets, tf.Tensor):
                    return_type = tf.Tensor
                elif isinstance(targets, list):
                    return_type = list
                else:
                    print(
                        "{} second return value of inputs should be a list or a tensor but is {}".
                        format(self._dataset.name, type(targets)))
                    return
            log_io(inputs)

            # Build a Graph that computes the predictions from the
            # inference model.
            # Preditions is an array of predictions with the same cardinality of
            # targets
            is_training_, predictions = self._model.get(
                inputs,
                self._dataset.num_classes,
                train_phase=True,
                l2_penalty=self._args["regularizations"]["l2"])
            if not isinstance(predictions, return_type):
                print((
                    "{} second return value must have the same type of the second"
                    "return value of inputs ({}) but is:").format(
                        self._model.name, return_type, type(predictions)))
                return

            num_of_parameters = count_trainable_parameters(print_model=True)
            print("Model {}: trainable parameters: {}. Size: {} KB".format(
                self._model.name, num_of_parameters, num_of_parameters * 4 /
                1000))

            # Calculate loss.
            loss = self._model.loss(predictions, targets)
            tf_log(tf.summary.scalar('loss', loss))

            # Create optimizer and log learning rate
            optimizer = builders.build_optimizer(self._args, self._steps,
                                                 global_step)
            train_op = optimizer.minimize(
                loss,
                global_step=global_step,
                var_list=variables_to_train(self._args["trainable_scopes"]))

            # TODO: more than 1 metric?

            train_metric = self._model.evaluator.metric["fn"](predictions,
                                                              targets)
            # General validation summary
            metric_value_ = tf.placeholder(tf.float32, shape=())
            metric_summary = tf.summary.scalar(
                self._model.evaluator.metric["name"], metric_value_)

            # read collection after that every op added its own
            # summaries in the train_summaries collection
            train_summaries = tf.summary.merge(
                tf.get_collection_ref(MODEL_SUMMARIES))

            # Build an initialization operation to run below.
            init = [
                tf.variables_initializer(tf.global_variables() +
                                         tf.local_variables()),
                tf.tables_initializer()
            ]

            # Start running operations on the Graph.
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                sess.run(init)

                # Start the queue runners with a coordinator
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Create the savers.
                train_saver, best_saver = builders.build_train_savers(
                    [global_step])
                flow.restore_or_restart(self._args, self._paths, sess,
                                        global_step)
                train_log, validation_log = builders.build_loggers(
                    sess.graph, self._paths)

                # If a best model already exists (thus we're continuing a train
                # process) then restore the best validation metric reached
                # and place it into best_metric_measured_value
                best_metric_measured_value = self._model.evaluator.eval(
                    self._paths["best"],
                    self._dataset,
                    input_type=InputType.validation,
                    batch_size=self._args["batch_size"])

                # Extract previous global step value
                old_gs = sess.run(global_step)

                # Restart from where we were
                for step in range(old_gs, self._steps["max"] + 1):
                    start_time = time.time()
                    _, loss_value = sess.run(
                        [train_op, loss], feed_dict={is_training_: True})

                    duration = time.time() - start_time

                    if np.isnan(loss_value):
                        print('Model diverged with loss = NaN')
                        break

                    # update logs every 10 iterations
                    if step % self._steps["log"] == 0:
                        examples_per_sec = self._args["batch_size"] / duration
                        sec_per_batch = float(duration)

                        format_str = ('{}: step {}, loss = {:.4f} '
                                      '({:.1f} examples/sec; {:.3f} sec/batch)')
                        print(
                            format_str.format(datetime.now(), step, loss_value,
                                              examples_per_sec, sec_per_batch))
                        # log train values
                        summary_lines = sess.run(
                            train_summaries, feed_dict={is_training_: True})
                        train_log.add_summary(summary_lines, global_step=step)

                    # Save the model checkpoint at the end of every epoch
                    # evaluate train and validation performance
                    if (step > 0 and step % self._steps["epoch"] == 0
                       ) or step == self._steps["max"]:
                        checkpoint_path = os.path.join(self._paths["log"],
                                                       'model.ckpt')
                        train_saver.save(
                            sess, checkpoint_path, global_step=step)

                        # validation metric
                        metric_measured_value = self._model.evaluator.eval(
                            self._paths["log"],
                            self._dataset,
                            input_type=InputType.validation,
                            batch_size=self._args["batch_size"])

                        summary_line = sess.run(
                            metric_summary,
                            feed_dict={metric_value_: metric_measured_value})
                        validation_log.add_summary(
                            summary_line, global_step=step)

                        # train metric
                        ta_value = sess.run(
                            train_metric, feed_dict={is_training_: False})
                        summary_line = sess.run(
                            metric_summary, feed_dict={metric_value_: ta_value})
                        train_log.add_summary(summary_line, global_step=step)

                        print(
                            '{} ({}): train {} = {:.3f} validation {} = {:.3f}'.
                            format(datetime.now(),
                                   int(step / self._steps["epoch"]),
                                   self._model.evaluator.metric["name"],
                                   ta_value, self._model.evaluator.metric[
                                       "name"], metric_measured_value))

                        # save best model
                        sign = math.copysign(
                            1,
                            metric_measured_value - best_metric_measured_value)
                        if sign == self._model.evaluator.metric[
                                "positive_trend_sign"]:
                            best_metric_measured_value = metric_measured_value
                            best_saver.save(
                                sess,
                                os.path.join(self._paths["best"], 'model.ckpt'),
                                global_step=step)
                # end of for
                validation_log.close()
                train_log.close()

                # When done, ask the threads to stop.
                coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)

            stats = self._model.evaluator.stats(
                self._paths["best"],
                self._dataset,
                batch_size=self._args["batch_size"])
            self._model.info = {
                "args": self._args,
                "paths": self._paths,
                "steps": self._steps,
                "stats": stats
            }
            return self._model.info
