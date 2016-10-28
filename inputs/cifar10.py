# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from . import utils

IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/cifar10_data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def read(filename_queue):
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
        "height": 32,
        "width": 32,
        "depth": 3,
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
    result["label"] = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.slice(record_bytes, [label_bytes], [image_bytes]),
        [result["depth"], result["height"], result["width"]])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    # RBG -> YUV
    #image = utils.rgb2yuv(image)

    # Subtract off the mean and divide by the variance of the pixels.
    result["image"] = tf.image.per_image_whitening(image)
    return result


def distorted_inputs(batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    filenames = [
        os.path.join(DATA_DIR, 'cifar-10-batches-bin/data_batch_%d.bin' % i)
        for i in range(1, 6)
    ]

    for name in filenames:
        if not tf.gfile.Exists(name):
            raise ValueError('Failed to find file: ' + name)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read(filename_queue)

    # Randomly flip the image horizontally, only.
    image = tf.image.random_flip_left_right(read_input["image"])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return utils.generate_image_and_label_batch(
        image,
        read_input["label"],
        min_queue_examples,
        batch_size,
        shuffle=True)


def inputs(eval_data, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    if not eval_data:
        filenames = [
            os.path.join(DATA_DIR, 'cifar-10-batches-bin/data_batch_%d.bin' % i)
            for i in range(1, 6)
        ]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [
            os.path.join(DATA_DIR, 'cifar-10-batches-bin/test_batch.bin')
        ]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for name in filenames:
        if not tf.gfile.Exists(name):
            raise ValueError('Failed to find file: ' + name)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

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
    """Download and extract the tarball from Alex's website."""
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
