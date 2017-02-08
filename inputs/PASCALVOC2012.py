#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
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
from . import utils
from .interfaces.Input import Input
from .interfaces.InputType import InputType


class PASCALVOC2012(Input):
    """Routine for decoding the PASCAL VOC 2012 binary file format."""

    def __init__(self):
        # Global constants describing the PASCAL VOC 2012 data set.
        # resize image to a fixed size
        # the resize dimension is an hyperparameter
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

        self._load_bboxes()

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

    def _read_image_and_boxes(self, filename_queue, input_type):
        """Extract the filename from the queue, read the image and extract
        from the lookup table the bboxes.
        Returns:
            image, bboxes
        """
        InputType.check(input_type)

        reader = tf.TextLineReader(skip_header_lines=False)
        _, filename = reader.read(filename_queue)

        # image is normalized in [-1,1]
        image = utils.read_image_jpg(
            tf.constant(
                os.path.join(self._data_dir, 'VOCdevkit', 'VOC2012',
                             'JPEGImages')) + "/" + filename + ".jpg")
        return image, self._tf_bboxes[str(input_type)].lookup(filename)

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

        # image is normalized in [-1,1]
        image = utils.read_image_jpg(image_path)
        return image, tf.stack([y_min, x_min, y_max, x_max, label])

    def distorted_inputs(self, batch_size):
        """Construct distorted input for PASCALVOC2012 training using the Reader ops.

        Args:
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, self._image_height, self._image_width, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size, 2 = angle + label]
        """

        filenames = [os.path.join(self._data_dir, 'train.csv')]

        for name in filenames:
            if not tf.gfile.Exists(name):
                raise ValueError('Failed to find file: ' + name)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self._num_examples_per_epoch_for_train *
                                 min_fraction_of_examples_in_queue)

        with tf.variable_scope("{}_input".format(InputType.train)):
            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer(filenames)

            image, bbox_and_label = self._read_image_and_box(
                filename_queue)  #bbox is a single box

            bbox = bbox_and_label[:4]
            label = bbox_and_label[4:]

            ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
            box_height = ymax - ymin
            box_width = xmax - xmin

            # expand bbbox by a 10%
            ymin = ymin * 1.05
            xmin = xmin * 0.95

            ymax = ymax * 0.95
            xmax = xmax * 1.05

            ymin = tf.minimum(ymin, ymax)
            ymax = tf.maximum(ymax, ymin)

            xmin = tf.minimum(xmin, xmax)
            xmax = tf.maximum(xmax, xmin)
            bbox = tf.clip_by_value(tf.stack([ymin, xmin, ymax, xmax]), 0., 1.)

            cropped_image = tf.squeeze(
                tf.image.crop_and_resize(
                    tf.expand_dims(image, axis=0),
                    tf.expand_dims(bbox, axis=0), [0],
                    [self._image_height, self._image_width]))

            # the bounding box to predict, should be a box that has the same
            # properties of the original box:
            # if the original box is vertical, the box should be vertical too
            # we just need to resize the box to be absolute to the cropped image
            # and not to the orignal image

            # TODO: if present in the dataset use it,
            # otherwise generate an approximative angle
            # TODO: or better, rotate the image by a random angle and
            # and try to predict it
            degree = tf.cond(
                tf.greater(box_height, box_width), lambda: tf.constant([90.]),
                lambda: tf.constant([0.]))

            # expand dims required to get a format [num_box = 1, degree, label]
            angle_and_label = tf.expand_dims(
                tf.concat([degree, label], axis=0), axis=0)

            # Generate a batch of images and labels by building up a queue of examples.
            return utils.generate_image_and_label_batch(
                cropped_image,
                angle_and_label,
                min_queue_examples,
                batch_size,
                shuffle=True)

    def inputs(self, input_type, batch_size):
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
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        with tf.variable_scope("{}_input".format(input_type)):
            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer(filenames)

            image, bboxes = self._read_image_and_boxes(filename_queue,
                                                       input_type)
            return utils.generate_image_and_label_batch(
                image, bboxes, min_queue_examples, batch_size, shuffle=False)

    def _load_bboxes(self):
        """load bboxes for every image"""

        with tf.variable_scope("load_bboxes"):
            max_bboxes = defaultdict(int)
            for current_set in ['train', 'val']:
                print(
                    'Building bounding box set for {}, it will take some time.'.
                    format(current_set))
                with open(
                        os.path.join(self._data_dir, '{}.csv'.format(
                            current_set))) as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=',')
                    for line in reader:
                        self._bboxes[current_set][line["filename"]].append(
                            tf.constant(
                                [[
                                    float(line["y_min"]),
                                    float(line["x_min"]),
                                    float(line["y_max"]),
                                    float(line["x_max"]),
                                    # cast label to float in order to place
                                    # label and coords in the same list
                                    int(line["label"])
                                ]],
                                dtype=tf.float32))

                max_bboxes[current_set] = 0
                for filename, bboxes in self._bboxes[current_set].items():
                    bboxes_len = len(bboxes)
                    if bboxes_len > max_bboxes[current_set]:
                        max_bboxes[current_set] = bboxes_len

                for filename, bboxes in self._bboxes[current_set].items():
                    bboxes_len = len(bboxes)
                    if bboxes_len < max_bboxes[current_set]:
                        for _ in range(0, max_bboxes[current_set] - bboxes_len):
                            self._bboxes[current_set][filename].append(
                                tf.constant([[-1., -1., -1., -1., -1.]]))

                filenames = tf.squeeze(
                    tf.stack(list(self._bboxes[current_set].keys())))
                bboxes = tf.reshape(
                    tf.squeeze(
                        tf.stack(list(self._bboxes[current_set].values()))),
                    [filenames.get_shape()[0].value, -1])

                default_value = tf.zeros_like(bboxes[0], dtype=tf.float32) - 1.
                self._tf_bboxes[
                    current_set] = tf.contrib.lookup.MutableHashTable(
                        key_dtype=tf.string,
                        value_dtype=tf.float32,
                        default_value=default_value)
                self._tf_bboxes[current_set].insert(filenames, bboxes)
                print('Done.')

    def _maybe_download_and_extract(self):
        """Download and extract the tarball"""
        dest_directory = self._data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self._data_url.split('/')[-1]
        archivepath = os.path.join(dest_directory, filename)
        if not os.path.exists(archivepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, float(count * block_size) /
                                  float(total_size) * 100.0))
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
        # Build train.csv and validation.csv file in self._data_dir
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
