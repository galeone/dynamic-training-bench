#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Class that defines a InputMeta and its hinner fields """

from functools import wraps
from .InputType import InputType


class InputMeta(type):
    """Class that defines a InputMeta and its hinner fields.
    This MetaClass wraps the inputs and distroted_inputs methods of every model,
    and made the input definitions unique per input type.
    It adds a private dictionary _queues of inputs to the concrete class
    """

    def __new__(cls, name, bases, attrs):
        attrs["_queues"] = {str(input_type): None for input_type in InputType}
        attrs["inputs"] = cls._single(attrs["inputs"])
        return super(InputMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def _single(cls, inputs):
        """Load only one input queue per type"""

        @wraps(inputs)
        def wrapper(self, *args, **kwargs):
            key = "input_type"
            if self._queues[str(kwargs[key])] is None:
                self._queues[str(kwargs[key])] = inputs(self, *args, **kwargs)
            return self._queues[str(kwargs[key])]

        return wrapper
