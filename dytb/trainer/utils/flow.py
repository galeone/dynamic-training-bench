#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utilities to control to flow execution of the trainers"""

import sys
import tensorflow as tf

from .builders import build_restore_saver


def restore_or_restart(args, paths, sess, global_step):
    """Restore actual session or restart the training.
    If SESS.checkpoint_path is setted, start a new train
    loading the weight from the lastest checkpoint in that path
    Args:
        sess: session
        paths: dict of paths
        global_step: global_step tensor
    """

    # first check if exists and checkpoint_path passed
    # from where to load the weights.
    # Return error if there's not
    pretrained_checkpoint = None
    if args["checkpoint_path"] != '':
        pretrained_checkpoint = tf.train.latest_checkpoint(
            args["checkpoint_path"])
        if not pretrained_checkpoint:
            print("[E] {} not valid".format(args["checkpoint_path"]))
            sys.exit(-1)

    if not args["force_restart"]:
        # continue training checkpoint
        continue_checkpoint = tf.train.latest_checkpoint(paths["log"])
        if continue_checkpoint:
            restore_saver = build_restore_saver(
                [global_step], scopes_to_remove=args["exclude_scopes"])
            restore_saver.restore(sess, continue_checkpoint)
        # else if the continue checkpoint does not exists
        # and the pretrained checkpoint has been specified
        # load the weights from the pretrained checkpoint
        elif pretrained_checkpoint:
            restore_saver = build_restore_saver(
                [], scopes_to_remove=args["exclude_scopes"])
            restore_saver.restore(sess, pretrained_checkpoint)
        else:
            print('[!] No checkpoint file found')
