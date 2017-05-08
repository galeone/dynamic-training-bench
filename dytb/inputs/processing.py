#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utils to dataset preprocessing"""

import os
import multiprocessing
import tensorflow as tf


def build_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor or a list of tensors like [label, attrA, ... ]
        min_queue_examples: int32, minimum number of samples to retain
           in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size containing the elements of labels
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = multiprocessing.cpu_count()
    if num_preprocess_threads > 2:
        num_preprocess_threads -= 2

    if isinstance(label, list):
        row = [image] + label
    else:
        row = [image, label]

    if shuffle:
        return tf.train.shuffle_batch(
            row,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    return tf.train.batch(
        row,
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)


def convert_to_tfrecords(dataset, name, data_dir):
    """ Converts the dataset in a TFRecord file with name.tfrecords.
    Save it into data_dir."""

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    if dataset.images.shape[0] != dataset.num_examples:
        raise ValueError('Images size {} does not match label size {}.'.format(
            dataset.images.shape[0], dataset.num_examples))
    rows = dataset.images.shape[1]
    cols = dataset.images.shape[2]
    depth = dataset.images.shape[3]

    filename = os.path.join(data_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(dataset.num_examples):
        image_raw = dataset.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(dataset.labels[index])),
                'image_raw': _bytes_feature(image_raw)
            }))
        writer.write(example.SerializeToString())
    writer.close()
