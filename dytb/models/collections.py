#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Collections"""

# name of the collection that holds non trainable
# but required variables for the current model
REQUIRED_NON_TRAINABLES = 'required_vars_collection'

# name of the collection that holds the summaries
# related to the model (and the train phase)
MODEL_SUMMARIES = 'model_summaries'

# losses collection
LOSSES = 'losses'
