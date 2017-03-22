#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Setup file to make dytb installable via pip"""

import io
import re
from setuptools import setup
from setuptools import find_packages

INIT_PY = io.open('dytb/__init__.py').read()
METADATA = dict(re.findall("__([a-z]+)__ = '([^']+)'", INIT_PY))
METADATA['doc'] = re.findall('"""(.+)"""', INIT_PY)[0]

setup(
    name='dytb',
    version=METADATA['version'],
    description=METADATA['doc'],
    author=METADATA['author'],
    author_email=METADATA['email'],
    url=METADATA['url'],
    download_url='/'.join((METADATA['url'].rstrip('/'), 'tarball',
                           METADATA['version'])),
    license='MPL',
    scripts=['scripts/dytb_evaluate', 'scripts/dytb_train'],
    packages=find_packages())
