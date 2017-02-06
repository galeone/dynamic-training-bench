#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
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


# Adapted from
# https://github.com/pavelgonchar/colornet/blob/master/train.py
def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
                                    [0.587, -0.331, -0.418],
                                    [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    rgb = tf.expand_dims(rgb, 0)

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    temp = tf.squeeze(temp, [0])

    return temp


# Adapted from
# https://github.com/pavelgonchar/colornet/blob/master/train.py
def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.multiply(yuv, 255)
    yuv2rgb_filter = tf.constant([[[[1., 1., 1.], [0., -0.34413999, 1.77199996],
                                    [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])

    yuv = tf.expand_dims(yuv, 0)
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp,
                      tf.multiply(
                          tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.divide(temp, 255)
    temp = tf.squeeze(temp, [0])
    return temp


def generate_image_and_label_batch(image, label, min_queue_examples, batch_size,
                                   shuffle):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
           in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = multiprocessing.cpu_count()
    if num_preprocess_threads > 2:
        num_preprocess_threads -= 2

    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, label_batch


def scale_image(image):
    """Returns the image tensor with values in [-1, 1].
    Args:
        image: [height, width, depth] tensor with values in [0,1]
    """
    image = tf.subtract(image, 0.5)
    # now image has values with zero mean in range [-0.5, 0.5]
    image = tf.multiply(image, 2.0)
    # now image has values with zero mean in range [-1, 1]
    return image


def read_image_jpg(image_path, depth=3):
    """Reads the image from image_path (tf.string tensor) [jpg image].
    Cast the result to float32 and scale it in [-1,1] (see scale_image)
    Reuturn:
        the decoded jpeg image, casted to float32
    """
    return scale_image(
        tf.image.convert_image_dtype(
            tf.image.decode_jpeg(tf.read_file(image_path), channels=depth),
            dtype=tf.float32))


def read_image_png(image_path, depth=3):
    """Reads the image from image_path (tf.string tensor) [jpg image].
    Cast the result to float32 and scale it in [-1,1] (see scale_image)
    Reuturn:
        the decoded jpeg image, casted to float32
    """
    return scale_image(
        tf.image.convert_image_dtype(
            tf.image.decode_png(tf.read_file(image_path), channels=depth),
            dtype=tf.float32))


def read_image(image_path, channel, image_type):
    """Wrapper around read_image_{jpg,png}"""
    if image_type == "jpg":
        image = read_image_jpg(image_path, channel)
    else:
        image = read_image_png(image_path, channel)
    return image


def convert_to_tfrecords(dataset, name, data_dir):
    """ Converts the dataset in a TFRecord file with name.tfrecords.
    Save it into data_dir."""

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    images = dataset.images
    labels = dataset.labels
    num_examples = dataset.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size {} does not match label size {}.'.format(
            images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(data_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height':
            _int64_feature(rows),
            'width':
            _int64_feature(cols),
            'depth':
            _int64_feature(depth),
            'label':
            _int64_feature(int(labels[index])),
            'image_raw':
            _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()
