#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
# Adapted from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_input.py
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Routine for decoding the CIFAR-10 binary file format."""

import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from ..processing import build_batch
from ..images import scale_image
from ..interfaces import Input, InputType


class Cifar10(Input):
    """Routine for decoding the CIFAR-10 binary file format."""

    def __init__(self):
        # Global constants describing the CIFAR-10 data set.
        self._name = 'CIFAR-10'
        self._image_height = 32
        self._image_width = 32
        self._image_depth = 3

        self._num_classes = 10
        self._num_examples_per_epoch_for_train = 50000
        self._num_examples_per_epoch_for_eval = 10000
        self._num_examples_per_epoch_for_test = self._num_examples_per_epoch_for_eval

        self._data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'Cifar10')
        self._data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
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

    @property
    def name(self):
        """Returns the name of the input source"""
        return self._name

    def _read(self, filename_queue):
        """Reads and parses examples from CIFAR10 data files.

      Recommendation: if you want N-way read parallelism, call this function
      N times.  This will give you N independent Readers reading different
      files & positions within those files, which will give better mixing of
      examples.

      Args:
        filename_queue: A queue of strings with the filenames to read from.

      Returns:
        An object representing a single example, with the following fields:
          height: number of rows in the result (32)
          width: number of columns in the result (32)
          depth: number of color channels in the result (3)
          key: a scalar string Tensor describing the filename & record number
            for this example.
          label: an int32 Tensor with the label in the range 0..9.
          image: a [height, width, depth] uint8 Tensor with the image data
      """

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        result = {
            "height": self._image_height,
            "width": self._image_width,
            "depth": self._image_depth,
            "label": None,
            "image": None
        }

        image_bytes = result["height"] * result["width"] * result["depth"]
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        label_bytes = 1  # 2 for CIFAR-100
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        result["label"] = tf.squeeze(
            tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32))

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.slice(record_bytes, [label_bytes], [image_bytes]),
            [result["depth"], result["height"], result["width"]])

        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        # Convert from [0, 255] -> [0, 1]
        image = tf.divide(image, 255.0)

        # Convert from [0, 1] -> [-1, 1]
        result["image"] = scale_image(image)

        return result

    def inputs(self, input_type, batch_size, augmentation_fn=None):
        """Construct input for CIFAR evaluation using the Reader ops.

        Args:
            input_type: InputType enum
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, self._image_height, self._image_width, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        InputType.check(input_type)

        if input_type == InputType.train:
            filenames = [
                os.path.join(self._data_dir,
                             'cifar-10-batches-bin/data_batch_%d.bin' % i)
                for i in range(1, 6)
            ]
            num_examples_per_epoch = self._num_examples_per_epoch_for_train
        else:
            filenames = [
                os.path.join(self._data_dir,
                             'cifar-10-batches-bin/test_batch.bin')
            ]
            num_examples_per_epoch = self._num_examples_per_epoch_for_eval

        for name in filenames:
            if not tf.gfile.Exists(name):
                raise ValueError('Failed to find file: ' + name)

        with tf.variable_scope("{}_input".format(input_type)):
            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer(filenames)

            # Read examples from files in the filename queue.
            read_input = self._read(filename_queue)
            if augmentation_fn:
                read_input["image"] = augmentation_fn(read_input["image"])

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(
                num_examples_per_epoch * min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of examples.
            return build_batch(
                read_input["image"],
                read_input["label"],
                min_queue_examples,
                batch_size,
                shuffle=input_type == InputType.train)

    def _maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""
        dest_directory = self._data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self._data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename,
                     float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(self._data_url, filepath,
                                                     _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size,
                  'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
