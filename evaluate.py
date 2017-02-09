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

from datetime import datetime
import math

import tensorflow as tf
import metrics
from inputs.interfaces.InputType import InputType
from models.utils import variables_to_restore
from models.interfaces.Autoencoder import Autoencoder
from models.interfaces.Classifier import Classifier
from models.interfaces.Regressor import Regressor
from CLIArgs import CLIArgs


def accuracy(checkpoint_path, model, dataset, input_type, batch_size=200):
    """
    Reads the checkpoint and use it to evaluate the model
    Args:
        checkpoint_path: checkpoint folder
        model: python package containing the model saved
        dataset: python package containing the dataset to use
        input_type: InputType enum, the input type of the input examples
        batch_size: batch size for the evaluation in batches
    Returns:
        average_accuracy: the average accuracy
    """
    InputType.check(input_type)

    with tf.Graph().as_default():
        # Get images and labels from the dataset
        with tf.device('/cpu:0'):
            images, labels = dataset.inputs(
                input_type=input_type, batch_size=batch_size)

        # Build a Graph that computes the predictions from the inference model.
        _, predictions = model.get(
            images, dataset.num_classes, train_phase=False)

        # Accuracy op
        accuracy = metrics.accuracy_op(predictions, labels)

        saver = tf.train.Saver(variables_to_restore())
        accuracy_value = 0.0
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
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
                # Counts the number of correct predictions.
                accuracy_sum = 0.0
                total_sample_count = num_iter * batch_size
                step = 0
                while step < num_iter and not coord.should_stop():
                    accuracy_sum += sess.run(accuracy)
                    step += 1

                accuracy_value = accuracy_sum / step
            except Exception as exc:
                coord.request_stop(exc)
            finally:
                coord.request_stop()

            coord.join(threads)
        return accuracy_value


def error(checkpoint_path, model, dataset, input_type, batch_size=200):
    """
    Reads the checkpoint and use it to evaluate the model
    Args:
        checkpoint_path: checkpoint folder
        model: python package containing the model saved
        dataset: python package containing the dataset to use
        input_type: InputType enum, the input type of the input examples
        batch_size: batch size for the evaluation in batches
    Returns:
        average_error: the average error
    """
    InputType.check(input_type)

    with tf.Graph().as_default():
        # Get images and labels from the dataset
        with tf.device('/cpu:0'):
            images, labels = dataset.inputs(
                input_type=input_type, batch_size=batch_size)

        # Build a Graph that computes the predictions from the inference model.
        if isinstance(model, Autoencoder):
            # Autoencoder does not use num_classes and the reconstruction is among
            # reconstructions (predictions) and images
            _, predictions = model.get(
                images, train_phase=False, l2_penalty=0.0)
            loss = model.loss(predictions, images)
        else:
            _, predictions = model.get(
                images, dataset.num_classes, train_phase=False, l2_penalty=0.0)
            loss = model.loss(predictions, labels)

        saver = tf.train.Saver(variables_to_restore())
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
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
                step = 0
                average_error = 0.0
                while step < num_iter and not coord.should_stop():
                    error_value = sess.run(loss)
                    step += 1
                    average_error += error_value

                average_error /= step
            except Exception as exc:
                coord.request_stop(exc)
            finally:
                coord.request_stop()

            coord.join(threads)
        return average_error


if __name__ == '__main__':
    ARGS, MODEL, DATASET = CLIArgs(
        description="Evaluate the model").parse_eval()

    INPUT_TYPE = InputType.test if ARGS.test else InputType.validation

    with tf.device(ARGS.eval_device):
        if isinstance(MODEL, Classifier):
            print('{}: {} accuracy = {:.3f}'.format(
                datetime.now(), 'test' if ARGS.test else 'validation',
                accuracy(
                    ARGS.checkpoint_path,
                    MODEL,
                    DATASET,
                    INPUT_TYPE,
                    batch_size=ARGS.batch_size)))

        if isinstance(MODEL, Autoencoder):
            print('{}: {} error = {:.3f}'.format(
                datetime.now(), 'test' if ARGS.test else 'validation',
                error(ARGS.checkpoint_path, MODEL, DATASET, INPUT_TYPE,
                      ARGS.batch_size)))

        if isinstance(MODEL, Regressor):
            print('{}: {} error = {:.3f}'.format(
                datetime.now(), 'test' if ARGS.test else 'validation',
                error(ARGS.checkpoint_path, MODEL, DATASET, INPUT_TYPE,
                      ARGS.batch_size)))
