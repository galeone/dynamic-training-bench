#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utility functions for model training and evaluation"""

# TODO: understand if metrics can be formalized using an abstract class
# or something different is better

from abc import ABCMeta, abstractproperty


class Metric(object, metaclass=ABCMeta):
    """Metric is a metric to measure and defined by its properties"""

    @staticmethod
    @abstractproperty
    def func(outputs, targets):
        """Metric to measure between outputs and targets"""

    @staticmethod
    @abstractproperty
    def name():
        """Name of the metric"""

    @staticmethod
    @abstractproperty
    def positive_trend_sign():
        """+1 or -1 depending on the expected trend when the
        metric goes well"""

    @staticmethod
    @abstractproperty
    def model_selection():
        """Boolean: true if the best model should be choosen looking
        at the trend of this metric"""

    @staticmethod
    @abstractproperty
    def average():
        """Boolean: true if the metric should be averaged over different
        measures among the dataset. If false the values are added"""

    @staticmethod
    @abstractproperty
    def tensorboard():
        """Boolean: True if the metric should be logged in Tensorboard.
        The metric should output a scalar to be logged"""
