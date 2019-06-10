# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import torch
import torch.nn as nn
import numpy as np

# clip gradient
def clip_gradient(optimizer, grad_clip):
  for group in optimizer.param_groups:
    for param in group['params']:
      if hasattr(param.grad, 'data'):
        param.grad.data.clamp_(-grad_clip, grad_clip)

# reset learning rate
def set_lr(optimizer, lr):
  for group in optimizer.param_groups:
    group['lr'] = lr
