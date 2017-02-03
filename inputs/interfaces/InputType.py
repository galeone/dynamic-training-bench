#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Enum to specify the data type requested"""

from enum import Enum, unique


@unique
class InputType(Enum):
    """Enum to specify the data type requested"""
    validation = 'validation'
    train = 'train'
    test = 'test'

    def __str__(self):
        """Return the string representation of the enum"""
        return self.value

    @staticmethod
    def check(input_type):
        """Check if input_type is an element of this Enum"""
        if not isinstance(input_type, InputType):
            raise ValueError("Invalid input_type, required a valid InputType")
