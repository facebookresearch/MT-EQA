# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import sys

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../pyutils/House3D'))
sys.path.insert(0, osp.join(this_dir, '../pyutils'))

# localized vqa
sys.path.insert(0, osp.join(this_dir, '..'))

# modular_vqa
sys.path.insert(0, osp.join(this_dir, '../nav'))
