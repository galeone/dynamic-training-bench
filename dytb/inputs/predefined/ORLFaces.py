#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""ORL Faces database input"""

import os
import sys
import zipfile
import glob
from PIL import Image

from six.moves import urllib
import tensorflow as tf
import numpy as np
from ..processing import convert_to_tfrecords, build_batch
from ..images import scale_image
from ..interfaces import Input, InputType


class ORLFaces(Input):
    """ORL Faces database input"""

    def __init__(self):
        # Global constants describing the ORL Faces data set.
        self._name = 'ORL-Faces'
        self._image_width = 92
        self._image_height = 112
        self._image_depth = 1

        self._num_classes = 40
        self._num_examples_per_epoch_for_train = 400
        self._num_examples_per_epoch_for_eval = 0
        self._num_examples_per_epoch_for_test = 0

        self._data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'ORLFaces')
        self._data_url = 'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip'
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
        # length IMAGE_WIDHT * self._image_height) to a uint8 tensor with
        # the same shape
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self._image_width * self._image_height])

        #`Reshape to a valid image
        image = tf.reshape(image, (self._image_height, self._image_width,
                                   self._image_depth))

        # Convert from [0, 255] -> [0, 1] floats.
        image = tf.divide(tf.cast(image, tf.float32), 255.0)

        # Convert from [0, 1] -> [-1, 1]
        result["image"] = scale_image(image)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        result["label"] = tf.cast(features['label'], tf.int32)

        return result

    def inputs(self, input_type, batch_size, augmentation_fn=None):
        """Construct input for ORL Faces evaluation using the Reader ops.

        Args:
            input_type: InputType enum.
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, self._image_width, self._image_height, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        InputType.check(input_type)

        with tf.variable_scope("{}_input".format(input_type)):
            filename = os.path.join(self._data_dir, 'faces.tfrecords')
            num_examples_per_epoch = self._num_examples_per_epoch_for_train

            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer([filename])

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
        """Download and extract the ORL Faces dataset"""

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
            with zipfile.ZipFile(filepath) as zip_f:
                zip_f.extractall(
                    os.path.join(dest_directory, filename.split('.')[-2]))

        # Convert to Examples and write the result to TFRecords.
        if not tf.gfile.Exists(os.path.join(self._data_dir, 'faces.tfrecords')):
            images = []
            labels = []

            for pgm in glob.glob("{}/*/*.pgm".format(
                    os.path.join(dest_directory, filename.split('.')[-2]))):
                images.append(
                    np.expand_dims(np.asarray(Image.open(pgm)), axis=2))
                labels.append(int(pgm.split("/")[-2].strip("s")))

            # Create dataset object
            dataset = lambda: None
            dataset.num_examples = self._num_examples_per_epoch_for_train
            dataset.images = np.array(images)
            dataset.labels = np.array(labels)
            convert_to_tfrecords(dataset, 'faces', self._data_dir)
