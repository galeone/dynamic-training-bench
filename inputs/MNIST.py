#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Routine for decoding the MNIST binary file format."""

import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
from . import utils
from .interfaces.Input import Input
from .interfaces.InputType import InputType


class MNIST(Input):
    """Routine for decoding the MNIST binary file format."""

    def __init__(self):
        # Global constants describing the MNIST data set.
        self._image_width = 28
        self._image_height = 28
        self._image_depth = 1
        mnist.IMAGE_PIXELS = self._image_width * self._image_height * self._image_depth

        self._num_classes = 10
        self._num_examples_per_epoch_for_train = 55000
        self._num_examples_per_epoch_for_eval = 5000
        self._num_examples_per_epoch_for_test = 10000

        self._data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'MNIST')
        self._maybe_download_and_extract()

    def num_examples(self, input_type):
        """Returns the number of examples per the specified input_type

        Args:
            input_type: InputType enum
        """
        InputType.check(input_type)

        if input_type == InputType.train:
            return self._num_examples_per_epoch_for_train
        elif input_type == InputType.test:
            return self._num_examples_per_epoch_for_test
        return self._num_examples_per_epoch_for_eval

    @property
    def num_classes(self):
        """Returns the number of classes"""
        return self._num_classes

    # adapted from:
    # https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
    def _read(self, filename_queue):
        """Reads and parses examples from MNIST data files.
        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.

        Args:
            filename_queue: A queue of strings with the filenames to read from.

        Returns:
          An object representing a single example, with the following fields:
              label: an int32 Tensor with the label in the range 0..9.
              image: a [height, width, depth] uint8 Tensor with the image data
        """

        result = {'image': None, 'label': None}

        reader = tf.TFRecordReader()
        _, value = reader.read(filename_queue)
        features = tf.parse_single_example(
            value,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                # int64 required
                'label': tf.FixedLenFeature([], tf.int64)
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([mnist.IMAGE_PIXELS])

        #`Reshape to a valid image
        image = tf.reshape(image, (self._image_height, self._image_width,
                                   self._image_depth))

        # Convert from [0, 255] -> [0, 1]
        image = tf.divide(tf.cast(image, tf.float32), 255.0)
        # Convert from [0, 1] -> [-1, 1]
        result["image"] = utils.scale_image(image)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        result["label"] = tf.cast(features['label'], tf.int32)
        return result

    def distorted_inputs(self, batch_size):
        """Construct distorted input for MNIST training using the Reader ops.

        Args:
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, self._image_width, self._image_height, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        with tf.variable_scope("{}_input".format(InputType.train)):
            # Create a queue that produces the filenames to read.
            filename = os.path.join(self._data_dir, 'train.tfrecords')
            filename_queue = tf.train.string_input_producer([filename])

            # Read examples from files in the filename queue.
            read_input = self._read(filename_queue)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(self._num_examples_per_epoch_for_train *
                                     min_fraction_of_examples_in_queue)
            print(
                ('Filling queue with {} MNIST images before starting to train. '
                 'This will take a few minutes.').format(min_queue_examples))

            # Generate a batch of images and labels by building up a queue of examples.
            return utils.generate_image_and_label_batch(
                read_input["image"],
                read_input["label"],
                min_queue_examples,
                batch_size,
                shuffle=True)

    def inputs(self, input_type, batch_size):
        """Construct input for MNIST evaluation using the Reader ops.

        Args:
            input_type: InputType enum.
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, self._image_width, self._image_height, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        InputType.check(input_type)

        if input_type == InputType.train:
            filename = os.path.join(self._data_dir, 'train.tfrecords')
            num_examples_per_epoch = self._num_examples_per_epoch_for_train
        elif input_type == InputType.validation:
            filename = os.path.join(self._data_dir, 'validation.tfrecords')
            num_examples_per_epoch = self._num_examples_per_epoch_for_eval
        elif input_type == InputType.test:
            filename = os.path.join(self._data_dir, 'test.tfrecords')
            num_examples_per_epoch = self._num_examples_per_epoch_for_test

        with tf.variable_scope("{}_input".format(input_type)):
            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer([filename])

            # Read examples from files in the filename queue.
            read_input = self._read(filename_queue)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return utils.generate_image_and_label_batch(
                read_input["image"],
                read_input["label"],
                min_queue_examples,
                batch_size,
                shuffle=False)

    def _maybe_download_and_extract(self):
        """Download and extract the MNIST dataset"""
        data_sets = mnist.read_data_sets(
            self._data_dir,
            dtype=tf.uint8,
            reshape=False,
            validation_size=self._num_examples_per_epoch_for_eval)

        # Convert to Examples and write the result to TFRecords.
        if not tf.gfile.Exists(os.path.join(self._data_dir, 'train.tfrecords')):
            utils.convert_to_tfrecords(data_sets.train, 'train', self._data_dir)

        if not tf.gfile.Exists(
                os.path.join(self._data_dir, 'validation.tfrecords')):
            utils.convert_to_tfrecords(data_sets.validation, 'validation',
                                       self._data_dir)

        if not tf.gfile.Exists(os.path.join(self._data_dir, 'test.tfrecords')):
            utils.convert_to_tfrecords(data_sets.test, 'test', self._data_dir)
