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
from ..models.visualization import log_images


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
        self._model.evaluator.dataset = dataset
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
            tf.set_random_seed(self._args["seed"])
            self._model.seed = self._args["seed"]
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Get inputs and targets: inputs is an input batch
            # target could be either an array of elements or a tensor.
            # it could be [label] or [label, attr1, attr2, ...]
            # or Tensor, where tensor is a standard tensorflow Tensor with
            # its own shape
            with tf.device('/cpu:0'):
                inputs, *targets = self._dataset.inputs(
                    input_type=InputType.train,
                    batch_size=self._args["batch_size"],
                    augmentation_fn=self._args["regularizations"][
                        "augmentation"]["fn"])

            # Build a Graph that computes the predictions from the
            # inference model.
            # Preditions is an array of predictions with the same cardinality of
            # targets
            is_training_, *predictions = self._model.get(
                inputs,
                self._dataset.num_classes,
                train_phase=True,
                l2_penalty=self._args["regularizations"]["l2"])

            if len(predictions) != len(targets):
                print(("{}.get 2nd return value and {}.inputs 2nd return "
                       "value must have the same cardinality but got: {} vs {}"
                      ).format(self._model.name, self._dataset.name,
                               len(predictions), len(targets)))
                return

            if len(predictions) == 1:
                predictions = predictions[0]
                targets = targets[0]

                # autoencoder, usually
                if predictions.shape == inputs.shape:
                    log_images("input_output", inputs, predictions)
                else:
                    log_images("input", inputs)
            else:
                log_images("input", inputs)

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

            model_selection_idx = -1
            # validation metrics arrays
            metrics_to_measure = []
            metric_values_ = []
            metric_summaries = []
            for idx, metric in enumerate(self._model.evaluator.metrics):
                if metric["model_selection"]:
                    model_selection_idx = idx

                if metric["tensorboard"]:
                    # Build tensorboard scalar visualizations using placeholder
                    metric_values_.append(tf.placeholder(tf.float32, shape=()))
                    metric_summaries.append(
                        tf.summary.scalar(metric["name"], metric_values_[idx]))

                    metrics_to_measure.append(metric)

            if model_selection_idx == -1:
                print(
                    "Please specify a metric in the evaluator with 'model_selection' not None"
                )
                return

            # visualizations
            visualizations_to_measure = []
            visualization_values_ = []
            visualization_summaries = []
            for idx, viz in enumerate(self._model.evaluator.visualizations):
                visualization_values_.append(
                    tf.placeholder(tf.float32, shape=None))
                visualization_summaries.append(
                    tf.summary.image(viz["name"], visualization_values_[idx]))
                visualizations_to_measure.append(viz)

            # read collection after that every op added its own
            # summaries in the train_summaries collection.
            # No metrics are addded to the MODEL_SUMMARIES collection
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
                # and place it into best_model_selection_measure
                best_model_selection_measure = self._model.evaluator.eval(
                    self._model.evaluator.metrics[model_selection_idx],
                    self._paths["best"],
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

                        # arrays of validation measures
                        validation_measured_metrics = []
                        # ta value is the model selection metric evaluate on the training
                        # set, useful just to output on the CLI
                        ta_value = 0
                        for idx, metric in enumerate(metrics_to_measure):
                            # validation metrics
                            validation_measured_metrics.append(
                                self._model.evaluator.eval(
                                    metric,
                                    self._paths["log"],
                                    input_type=InputType.validation,
                                    batch_size=self._args["batch_size"]))
                            validation_log.add_summary(
                                sess.run(
                                    metric_summaries[idx],
                                    feed_dict={
                                        metric_values_[idx]:
                                        validation_measured_metrics[idx]
                                    }),
                                global_step=step)

                            # Repeat measurement on the training set
                            measure = self._model.evaluator.eval(
                                metric,
                                self._paths["log"],
                                input_type=InputType.train,
                                batch_size=self._args["batch_size"])
                            train_log.add_summary(
                                sess.run(
                                    metric_summaries[idx],
                                    feed_dict={metric_values_[idx]: measure}),
                                global_step=step)

                            # fill ta_value
                            if idx == model_selection_idx:
                                ta_value = measure

                        # visualization
                        for idx, viz in enumerate(visualizations_to_measure):
                            # validation metrics
                            measured_viz = self._model.evaluator.visualize(
                                viz,
                                self._paths["log"],
                                input_type=InputType.validation,
                                batch_size=self._args["batch_size"])
                            validation_log.add_summary(
                                sess.run(
                                    visualization_summaries[idx],
                                    feed_dict={
                                        visualization_values_[idx]: measured_viz
                                    }),
                                global_step=step)

                            # Repeat measurement on the training set
                            measured_viz = self._model.evaluator.visualize(
                                viz,
                                self._paths["log"],
                                input_type=InputType.train,
                                batch_size=self._args["batch_size"])
                            train_log.add_summary(
                                sess.run(
                                    visualization_summaries[idx],
                                    feed_dict={
                                        visualization_values_[idx]: measured_viz
                                    }),
                                global_step=step)

                        name = self._model.evaluator.metrics[
                            model_selection_idx]["name"]

                        print(
                            '{} ({}): train {} = {:.3f} validation {} = {:.3f}'.
                            format(datetime.now(),
                                   int(step / self._steps["epoch"]), name,
                                   ta_value, name, validation_measured_metrics[
                                       model_selection_idx]))

                        # save best model
                        sign = math.copysign(
                            1, validation_measured_metrics[model_selection_idx]
                            - best_model_selection_measure)
                        if sign == self._model.evaluator.metrics[
                                model_selection_idx]["positive_trend_sign"]:
                            best_model_selection_measure = validation_measured_metrics[
                                model_selection_idx]
                            best_saver.save(
                                sess,
                                os.path.join(self._paths["best"], 'model.ckpt'),
                                global_step=step)
                        # end of metrics

                        # end of visualizations
                        # end of for
                validation_log.close()
                train_log.close()

                # When done, ask the threads to stop.
                coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)

            stats = self._model.evaluator.stats(
                self._paths["best"], batch_size=self._args["batch_size"])
            self._model.info = {
                "args": self._args,
                "paths": self._paths,
                "steps": self._steps,
                "stats": stats
            }
            return self._model.info
