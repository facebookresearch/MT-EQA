import collections
import torch
import torch.nn as nn


def clip_gradient(optimizer, grad_clip):
  for group in optimizer.param_groups:
    for param in group['params']:
      param.grad.data.clamp_(-grad_clip, grad_clip)

def set_lr(optimizer, lr):
  for group in optimizer.param_groups:
    group['lr'] = lr