#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate Classification models """

from .Evaluator import Evaluator
from .metrics import accuracy_op, confusion_matrix_op


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
