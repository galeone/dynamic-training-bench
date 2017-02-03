#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Class that defines a Model and its hinner fields """

from functools import wraps


class Model(type):
    """Class that defines a Model and its hinner fields.
    This MetaClass wraps the get method of every model,
    and made the model definition in train and evaluation mode
    run only once.
    It adds the _trainable_model and _evaluation_model fields
    to the concrete class.
    """

    def __new__(cls, name, bases, attrs):
        attrs["_trainable_model"], attrs["_evaluation_model"] = None, None
        attrs["get"] = cls._single(attrs["get"])
        return super(Model, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def _single(cls, get):
        """Annotation to use when getting a model.
        The model is instantiated once per train phase and eval phase.
        In this way we have at most 2 models in memory"""

        @wraps(get)
        def wrapper(self, *args, **kwargs):
            if kwargs["train_phase"]:
                if self._trainable_model is None:
                    self._trainable_model = get(self, *args, **kwargs)
                return self._trainable_model

            if self._evaluation_model is None:
                self._evaluation_model = get(self, *args, **kwargs)
            return self._evaluation_model

        return wrapper
