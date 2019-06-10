# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import time
import argparse
import random
import numpy as np
import logging
import os, sys, json
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _init_paths
from nav.loaders.nav_imitation_loader import NavImitationDataset
from nav.models.navigator import Navigator
from nav.models.crits import SeqModelCriterion, MaskedMSELoss
import nav.models.utils as model_utils

import tensorboardX as tb


def clip_model_gradient(params, grad_clip):
  for param in params:
    if hasattr(param.grad, 'data'):
      param.grad.data.clamp_(-grad_clip, grad_clip)

def ensure_shared_grads(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad

def imitation(rank, args, shared_nav_model, counter):
  # set up tensorboard
  writer = tb.SummaryWriter(args.tb_dir, filename_suffix=str(rank))

  # set up cuda device
  torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

  # set up random seeds
  random.seed(args.seed + rank)
  np.random.randn(args.seed + rank)
  torch.manual_seed(args.seed + rank)

  # set up loader
  train_loader_kwargs = {
    'data_json': args.imitation_data_json,
    'data_h5': args.imitation_data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'train',
    'max_seq_length': args.max_seq_length,
    'requires_imgs': False,
    'nav_types': args.nav_types,
    'question_types': ['all'],
  }
  train_dataset = NavImitationDataset(**train_loader_kwargs)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1)
  print('train_loader set up.')

  # set up optimizer on shared_nav_model
  # lr = 5e-5
  lr = args.learning_rate
  optimizer = torch.optim.Adam(shared_nav_model.parameters(), lr=lr, 
                               betas=(args.optim_alpha, args.optim_beta), eps=args.optim_epsilon,
                               weight_decay=args.weight_decay)

  # set up models
  opt = vars(args)
  opt['act_to_ix'] = train_dataset.act_to_ix
  opt['num_actions'] = len(opt['act_to_ix'])
  model = Navigator(opt)
  model.cuda()
  print('navigator set up.')

  # set up criterions
  nll_crit = SeqModelCriterion().cuda()

  # -
  epoch = 0
  iters = 0

  # train
  while True:

    for batch in train_loader:
      # sync model
      model.load_state_dict(shared_nav_model.state_dict())
      model.train()
      model.cuda()

      # batch = {qid, path_ix, house, id, type, phrase, phrase_emb, ego_feats, next_feats, res_feats,
      #  action_inputs, action_outputs, action_masks, ego_imgs}
      ego_feats = batch['ego_feats'].cuda()  # (n, L, 3200)
      phrase_embs = batch['phrase_emb'].cuda()  # (n, 300)
      action_inputs = batch['action_inputs'].cuda()   # (n, L)
      action_outputs = batch['action_outputs'].cuda() # (n, L)
      action_masks = batch['action_masks'].cuda()  # (n, L)
      # forward
      # - logprobs (n, L, #actions)
      # - output_feats (n, L, rnn_size)
      # - pred_feats (n, L, 3200) or None
      logprobs, _, pred_feats, _ = model(ego_feats, phrase_embs, action_inputs)  
      nll_loss = nll_crit(logprobs, action_outputs, action_masks)

      # backward
      optimizer.zero_grad()
      nll_loss.backward()
      clip_model_gradient(model.parameters(), args.grad_clip)
      ensure_shared_grads(model.cpu(), shared_nav_model)
      optimizer.step()

      if iters % 25 == 0:
        print('imitation-r%s(ep%s it%s lr%.2E loss%.4f)' % (rank, epoch, iters, lr, nll_loss))

      # write to tensorboard
      writer.add_scalar('imitation_rank/nll_loss', float(nll_loss.item()), counter.value)

      # increate iters
      iters += 1

      # decay learning rate
      if args.lr_decay > 0:
        if args.im_learning_rate_decay_start > 0 and iters > args.im_learning_rate_decay_start:
          frac = (iters - args.im_learning_rate_decay_start) / args.im_learning_rate_decay_every
          decay_factor = 0.1 ** frac
          lr = args.learning_rate * decay_factor
          model_utils.set_lr(optimizer, lr)
    
    epoch += 1
      
