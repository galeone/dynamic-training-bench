#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""PASCAL VOC 2012"""

import os
import sys
import tarfile
import xml.etree.ElementTree as etree
import csv
from collections import defaultdict

from six.moves import urllib
import tensorflow as tf
from ..processing import build_batch
from ..images import read_image_jpg
from ..interfaces import Input, InputType
from ..PASCALVOC2012Classification import PASCALVOC2012Classification


class PASCALVOC2012Localization(Input):
    """Routine for decoding the PASCAL VOC 2012 binary file format."""

    def __init__(self):
        self._name = 'PASCAL-VOC-2012-Localization'
        # multiple boxes enable the return of a tensor
        # of boxes instead of a single box per image
        self._multiple_bboxes = False

        # Use Classification dataset
        # to extract shared features and download the dataset
        self._pascal = PASCALVOC2012Classification()

    def num_examples(self, input_type):
        """Returns the number of examples per the specified input_type

        Args:
            input_type: InputType enum
        """
        return self._pascal.num_examples(input_type)

    @property
    def num_classes(self):
        """Returns the number of classes"""
        return self._pascal.num_classes

    @property
    def name(self):
        """Returns the name of the input source"""
        return self._name

    def _read_image_and_box(self, bboxes_csv):
        """Extract the filename from the queue, read the image and
        produce a single box
        Returns:
            image, [y_min, x_min, y_max, x_max, label]
        """

        reader = tf.TextLineReader(skip_header_lines=True)
        _, row = reader.read(bboxes_csv)
        # file ,y_min, x_min, y_max, x_max, label
        record_defaults = [[""], [0.], [0.], [0.], [0.], [0.]]
        # eg:
        # 2008_000033,0.1831831831831832,0.208,0.7717717717717718,0.952,0
        filename, y_min, x_min, y_max, x_max, label = tf.decode_csv(
            row, record_defaults)
        image_path = os.path.join(self._data_dir, 'VOCdevkit', 'VOC2012',
                                  'JPEGImages') + "/" + filename + ".jpg"

        # image is normalized in [-1,1]
        image = read_image_jpg(image_path)
        return image, tf.stack([y_min, x_min, y_max, x_max, label])

    def inputs(self, input_type, batch_size, augmentation_fn=None):
        """Construct input for PASCALVOC2012 evaluation using the Reader ops.

        Args:
            input_type: InputType enum
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, self._image_height, self._image_width, self._image_depth] size.
            labels: A tensor with shape [batch_size, num_bboxes_max, 5]. num_bboxes_max are the maximum bboxes found in the
            requested set (train/test/validation). Where the bbox is fake, a -1,-1,-1,-1,-1 value is present
        """
        InputType.check(input_type)

        if input_type == InputType.train:
            filenames = [
                os.path.join(self._data_dir, 'VOCdevkit', 'VOC2012',
                             'ImageSets', 'Main', 'train.txt')
            ]
            num_examples_per_epoch = self._num_examples_per_epoch_for_train
        else:
            filenames = [
                os.path.join(self._data_dir, 'VOCdevkit', 'VOC2012',
                             'ImageSets', 'Main', 'val.txt')
            ]
            num_examples_per_epoch = self._num_examples_per_epoch_for_eval

        for name in filenames:
            if not tf.gfile.Exists(name):
                raise ValueError('Failed to find file: ' + name)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(
            num_examples_per_epoch * min_fraction_of_examples_in_queue)

        with tf.variable_scope("{}_input".format(input_type)):
            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer(filenames)

            image, bbox = self._read_image_and_box(filename_queue)

            if augmentation_fn:
                image = augmentation_fn(image)
            return build_batch(
                image,
                bbox,
                min_queue_examples,
                batch_size,
                shuffle=input_type == InputType.train)
