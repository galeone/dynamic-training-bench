#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate Classification models """

import numpy as np
import tensorflow as tf
from .Evaluator import Evaluator
from .metrics import accuracy_op, confusion_matrix_op
from ..models.utils import variables_to_restore


class ClassifierEvaluator(Evaluator):
    """ClassifierEvaluator is the evaluation object for a Classifier model"""

    @property
    def metrics(self):
        """Returns a list of dict with keys:
        {
            "fn": function
            "name": name
            "positive_trend_sign": sign that we like to see when things go well
            "model_selection": boolean, True if the metric has to be measured to select the model
            "average": boolean, true if the metric should be computed as average over the batches.
                       If false the results over the batches are just added
            "tensorboard": boolean. True if the metric is a scalar and can be logged in tensoboard
        }
        """
        return [{
            "fn": accuracy_op,
            "name": "accuracy",
            "positive_trend_sign": +1,
            "model_selection": True,
            "average": True,
            "tensorboard": True,
        }, {
            "fn":
            lambda logits, labels: confusion_matrix_op(logits, labels, self.dataset.num_classes),
            "name":
            "confusion_matrix",
            "positive_trend_sign":
            0,
            "model_selection":
            False,
            "average":
            False,
            "tensorboard":
            False,
        }]

    def extract_features(self, checkpoint_path, inputs, layer_name,
                         num_classes):
        """Restore model parameters from checkpoint_path. Search in the model
        the layer with name `layer_name`. If found places `inputs` as input to the model
        and returns the values extracted by the layer.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            inputs: a Tensor with a shape compatible with the model's input
            layer_name: a string, the name of the layer to extract from model
            num_classes: number of classes to classify, this number must be equal to the number
            of classes the classifier was trained on, otherwise the restore from checkpoint fails
        Returns:
            features: a numpy ndarray that contains the extracted features
        """

        # Evaluate the inputs in the current default graph
        # then user a placeholder to inject the computed values into the new graph
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            evaluated_inputs = sess.run(inputs)

        # Create a new graph to not making dirty the default graph after subsequent
        # calls
        with tf.Graph().as_default() as graph:
            inputs_ = tf.placeholder(inputs.dtype, shape=inputs.shape)

            # Build a Graph that computes the predictions from the inference model.
            _ = self._model.get(inputs_, num_classes, train_phase=False)

            # This will raise an exception if layer_name is not found
            layer = graph.get_tensor_by_name(layer_name)

            saver = tf.train.Saver(variables_to_restore())
            features = np.zeros(layer.shape)
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True)) as sess:
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('[!] No checkpoint file found')
                    return features

                features = sess.run(
                    layer, feed_dict={inputs_: evaluated_inputs})

            return features
