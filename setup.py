#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Setup file to make dytb installable via pip"""

from setuptools import setup
from setuptools import find_packages

setup(
    name='dytb',
    version='1.0.0',
    description='Simplify the trainining and tuning of Tensorflow models',
    author='Paolo Galeone',
    author_email='nessuno@nerdz.eu',
    url='https://github.com/galeone/dynamic-training-bench',
    download_url='https://github.com/galeone/dynamic-training-bench/tarball/1.0.0',
    license='MPL',
    packages=find_packages())
