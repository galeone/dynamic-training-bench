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
from ..interfaces.Input import Input
from ..interfaces.InputType import InputType


class PASCALVOC2012Classification(Input):
    """Routine for decoding the PASCAL VOC 2012 binary file format."""

    def __init__(self):
        # Global constants describing the PASCAL VOC 2012 data set.
        # resize image to a fixed size
        # the resize dimension is an hyperparameter
        self._name = 'PASCAL-VOC-2012-Classification'
        self._image_height = 150
        self._image_width = 150
        self._image_depth = 3

        # multiple boxes enable the return of a tensor
        # of boxes instead of a single box per image
        self._multiple_bboxes = False

        self.CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self._bboxes = {"train": defaultdict(list), "val": defaultdict(list)}
        self._tf_bboxes = {"train": None, "val": None}
        self._num_classes = 20
        self._num_examples_per_epoch_for_train = 13609
        self._num_examples_per_epoch_for_eval = 13841
        self._num_examples_per_epoch_for_test = self._num_examples_per_epoch_for_eval

        self._data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'PASCALVOC2012')
        self._data_url = 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
        self._maybe_download_and_extract()

    @property
    def name(self):
        """Returns the name of the input source"""
        return self._name

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

    def _read_image_and_box(self, bboxes_csv):
        """Extract the filename from the queue, read the image and
        produce a single box
        Returns:
            image, box
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

        # image is normalized in [-1,1], convert to #_image_depth depth
        image = read_image_jpg(image_path, depth=self._image_depth)
        return image, tf.stack([y_min, x_min, y_max, x_max, label])

    def _read(self, filename_queue):
        image, bbox_and_label = self._read_image_and_box(
            filename_queue)  #bbox is a single box

        bbox = bbox_and_label[:4]
        label = tf.cast(bbox_and_label[-1], tf.int32)

        image = tf.squeeze(
            tf.image.crop_and_resize(
                tf.expand_dims(image, axis=0),
                tf.expand_dims(bbox, axis=0),
                box_ind=[0],
                crop_size=[self._image_height, self._image_width]),
            axis=[0])
        return image, label

    def inputs(self, input_type, batch_size, augmentation_fn=None):
        """Construct input for PASCALVOC2012Classification evaluation using the Reader ops.

        Args:
            input_type: InputType enum
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, self._image_height, self._image_width, self._image_depth] size.
            labels: tensor with batch_size labels
        """
        InputType.check(input_type)

        if input_type == InputType.train:
            filenames = [os.path.join(self._data_dir, 'train.csv')]
            num_examples_per_epoch = self._num_examples_per_epoch_for_train
        else:
            filenames = [os.path.join(self._data_dir, 'val.csv')]
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

            image, label = self._read(filename_queue)
            if augmentation_fn:
                image = augmentation_fn(image)

            return build_batch(
                image,
                label,
                min_queue_examples,
                batch_size,
                shuffle=input_type == InputType.train)

    def _maybe_download_and_extract(self):
        """Download and extract the tarball"""
        dest_directory = self._data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self._data_url.split('/')[-1]
        archivepath = os.path.join(dest_directory, filename)
        if not os.path.exists(archivepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename,
                     float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            archivepath, _ = urllib.request.urlretrieve(self._data_url,
                                                        archivepath, _progress)
            print()
            statinfo = os.stat(archivepath)
            print('Successfully downloaded', filename, statinfo.st_size,
                  'bytes.')
            tarfile.open(archivepath, 'r').extractall(dest_directory)
            print('Sucessfully extracted.')

        # Now self._data dir contains VOCDevkit folder
        # Build train.csv and val.csv file in self._data_dir
        csv_header = ["filename", "y_min", "x_min", "y_max", "x_max", "label"]
        if os.path.exists(
                os.path.join(self._data_dir, 'train.csv')) and os.path.exists(
                    os.path.join(self._data_dir, 'val.csv')):
            return

        base_dir = os.path.join(
            self._data_dir,
            'VOCdevkit',
            'VOC2012',)

        for current_set in ['train', 'val']:
            csv_path = os.path.join(self._data_dir,
                                    '{}.csv'.format(current_set))
            with open(csv_path, mode='w') as csv_file:
                # header
                writer = csv.DictWriter(csv_file, csv_header)
                writer.writeheader()
                for current_class in self.CLASSES:
                    lines = open(
                        os.path.join(base_dir, 'ImageSets', 'Main', '{}_{}.txt'.
                                     format(current_class, current_set))).read(
                                     ).strip().split("\n")
                    for line in lines:
                        splitted = line.split()
                        if len(splitted) < 1:
                            print(splitted, line, current_class)
                        if splitted[1] == "-1":
                            continue

                        image_xml = os.path.join(base_dir, 'Annotations',
                                                 '{}.xml'.format(splitted[0]))
                        image_filename = splitted[0]

                        # parse XML
                        tree = etree.parse(image_xml)
                        root = tree.getroot()
                        size = root.find('size')
                        width = float(size.find('width').text)
                        height = float(size.find('height').text)

                        for obj in root.iter('object'):
                            # skip difficult & object.name not in current class
                            label = obj.find('name').text
                            if label != current_class:
                                continue

                            difficult = obj.find('difficult').text
                            if int(difficult) == 1:
                                continue

                            bndbox = obj.find('bndbox')
                            normalized_bbox = [
                                # y_min
                                float(bndbox.find('ymin').text) / height,
                                # x_min
                                float(bndbox.find('xmin').text) / width,
                                # y_max
                                float(bndbox.find('ymax').text) / height,
                                # x_max
                                float(bndbox.find('xmax').text) / width
                            ]

                            label_id = self.CLASSES.index(current_class)
                            writer.writerow({
                                "filename": image_filename,
                                "y_min": normalized_bbox[0],
                                "x_min": normalized_bbox[1],
                                "y_max": normalized_bbox[2],
                                "x_max": normalized_bbox[3],
                                "label": label_id
                            })
            print('{}.csv created'.format(current_set))
