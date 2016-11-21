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

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_DEPTH = 1
mnist.IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_DEPTH

# Global constants describing the MNIST data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 55000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10000

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/MNIST_data"


# adapted from:
# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
def read(filename_queue):
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
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    result["image"] = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    result["label"] = tf.cast(features['label'], tf.int32)
    return result


def distorted_inputs(batch_size):
    """Construct distorted input for MNIST training using the Reader ops.

    Args:
        batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    # Create a queue that produces the filenames to read.
    filename = os.path.join(DATA_DIR, 'train.tfrecords')
    filename_queue = tf.train.string_input_producer([filename])

    # Read examples from files in the filename queue.
    read_input = read(filename_queue)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print(('Filling queue with {} MNIST images before starting to train. '
           'This will take a few minutes.').format(min_queue_examples))

    # Generate a batch of images and labels by building up a queue of examples.
    return utils.generate_image_and_label_batch(
        read_input["image"],
        read_input["label"],
        min_queue_examples,
        batch_size,
        shuffle=True)


def inputs(input_type, batch_size):
    """Construct input for MNIST evaluation using the Reader ops.

    Args:
        input_type: Type enum.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    if not isinstance(input_type, utils.Type):
        raise ValueError("Invalid input_type, required a valid Type")

    if input_type == utils.Type.train:
        filename = os.path.join(DATA_DIR, 'train.tfrecords')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    elif input_type == utils.Type.validation:
        filename = os.path.join(DATA_DIR, 'validation.tfrecords')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    elif input_type == utils.Type.test:
        filename = os.path.join(DATA_DIR, 'test.tfrecords')
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([filename])

    # Read examples from files in the filename queue.
    read_input = read(filename_queue)

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


def maybe_download_and_extract():
    """Download and extract the MNIST dataset"""
    data_sets = mnist.read_data_sets(
        DATA_DIR,
        dtype=tf.uint8,
        reshape=False,
        validation_size=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)

    # Convert to Examples and write the result to TFRecords.
    if not tf.gfile.Exists(os.path.join(DATA_DIR, 'train.tfrecords')):
        utils.convert_to_tfrecords(data_sets.train, 'train', DATA_DIR)

    if not tf.gfile.Exists(os.path.join(DATA_DIR, 'validation.tfrecords')):
        utils.convert_to_tfrecords(data_sets.validation, 'validation', DATA_DIR)

    if not tf.gfile.Exists(os.path.join(DATA_DIR, 'test.tfrecords')):
        utils.convert_to_tfrecords(data_sets.test, 'test', DATA_DIR)
