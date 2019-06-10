# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .StackRNNCell import StackedRNN, StackedLSTM

"""
Navigator
"""
class Navigator(nn.Module):

  def __init__(self, opt):
    super(Navigator, self).__init__()
    # options
    self.rnn_type = opt.get('rnn_type', 'lstm')
    self.rnn_size = opt.get('rnn_size', 256)
    self.num_layers = opt.get('num_layers', 1)
    self.rnn_dropout = opt.get('rnn_dropout', 0.2)  # rnn dropout

    self.seq_dropout = opt.get('seq_dropout', 0.) # dropout for rnn output
    self.fc_dim = opt.get('fc_dim', 64)
    self.act_dim = opt.get('act_dim', 64)
    self.fc_dropout = opt.get('fc_dropout', 0.0)  # dropout after fc layer
    self.num_input_actions = opt.get('num_actions', 5) # {0:'forward', 1:'left', 2:'right', 3:'stop', 4:'dummy'}
    self.num_output_actions = self.num_input_actions-1 # {0:'forward', 1:'left', 2:'right', 3:'stop'}

    self.use_action = opt.get('use_action', True)      # use action as input
    self.use_residual = opt.get('use_residual', False) # predict residual
    self.use_next = opt.get('use_next', False)         # predict next image feature
    assert (self.use_residual and self.use_next) == False, 'use_residual and use_next cannot be both true.'

    # img embedding
    self.cnn_fc_layer = nn.Sequential(nn.Linear(3200, self.fc_dim), nn.ReLU(), nn.Dropout(p=self.fc_dropout))

    # phrase embedding 
    self.phrase_embed = nn.Sequential(nn.Linear(300, self.fc_dim), nn.ReLU(), nn.Dropout(p=self.fc_dropout))

    # action embedding
    self.action_embed = nn.Embedding(self.num_input_actions, self.act_dim)

    # create core rnn
    self.rnn_input_size = self.fc_dim
    if self.use_action: self.rnn_input_size += self.act_dim  # we follow eqa_v1, use concat here!
    if self.rnn_type in ['rnn', 'gru']:
      self.core = StackedRNN(self.num_layers, self.rnn_input_size, self.rnn_size, self.rnn_dropout, self.rnn_type)
    else:
      self.core = StackedLSTM(self.num_layers, self.rnn_input_size, self.rnn_size, self.rnn_dropout)

    # generator
    self.act_gen = nn.Sequential(nn.Linear(self.rnn_size, self.fc_dim), 
                                 nn.ReLU(), 
                                 nn.Dropout(p=self.seq_dropout),
                                 nn.Linear(self.fc_dim, self.num_output_actions),
                                 nn.LogSoftmax(dim=1))
    self.feat_gen = None
    if self.use_residual or self.use_next:
      # next_feat (maybe res_feat also) would contain lots of zeros after ReLU() layer in CNN
      # we mimic this process
      self.feat_gen = nn.Sequential(nn.Linear(self.rnn_size, self.fc_dim), 
                                    nn.ReLU(),
                                    nn.Dropout(p=self.seq_dropout),
                                    nn.Linear(self.fc_dim, 3200),
                                    nn.ReLU())

  def init_hidden(self, batch_size):
    """first hidden vector(s)"""
    weight = next(self.parameters()).data
    if self.rnn_type == 'lstm':
      return (weight.new(self.num_layers, batch_size, self.rnn_size).zero_(),
              weight.new(self.num_layers, batch_size, self.rnn_size).zero_())
    else:
      return weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
  
  def forward(self, img_feats, phrase_embs, action_inputs):
    """
    Inputs:
    - img_feats     (n, L, 3200) float
    - phrase_embs   (n, 300) float
    - action_inputs (n, L) int
    Outputs:
    - logprobs      (n, L, #actions) float
    - output_feats  (n, L, rnn_size) float
    - pred_feats    (n, L, 3200) float / None
    - state
    """
    batch_size, L = img_feats.shape[0], img_feats.shape[1]

    # initialize
    img_feats = self.cnn_fc_layer(img_feats)     # (n, L, fc_dim)
    emb_feats = self.phrase_embed(phrase_embs)   # (n, fc_dim)
    act_feats = self.action_embed(action_inputs) # (n, L, fc_dim)
    state = self.init_hidden(batch_size)
    logprobs = []
    output_feats = []
    pred_feats = []
    # forward
    for t in range(L):
      img_t = img_feats[:, t, :]  # (n, fc_dim)
      emb_t = emb_feats           # (n, fc_dim)
      act_t = act_feats[:, t, :]  # (n, fc_dim)
      fuse_t = torch.mul(img_t, emb_t)
      if self.use_action:
        fuse_t = torch.cat([fuse_t, act_t], 1)  # (n, fc_dim * 2)
      # rnn      
      output_t, state = self.core(fuse_t, state)  # (n, rnn_size)
      # generate
      logprobs_t = self.act_gen(output_t)        # (n, #actions)
      logprobs.append(logprobs_t.unsqueeze(1))   # (n, 1, #actions)
      output_feats.append(output_t.unsqueeze(1)) # (n, 1, rnn_size)
      if self.feat_gen:
        pred_feats_t = self.feat_gen(output_t)        # (n, 3200)
        pred_feats.append(pred_feats_t.unsqueeze(1))  # (n, 1, 3200)
    # output
    logprobs = torch.cat(logprobs, 1)  # (n, L, #actions)
    output_feats = torch.cat(output_feats, 1)  # (n, L, rnn_size)
    if self.feat_gen:
      pred_feats = torch.cat(pred_feats, 1)  # (n, L, 3200)
    return logprobs, output_feats, pred_feats, state
  
  def forward_step(self, img_feat, phrase_emb, action_input, state=None):
    """
    Inputs:
    - img_feat      (n, 3200) float
    - phrase_emb    (n, 300) float
    - action_input  (n, ) int
    - state         tuple of hidden and cell states
    Output:
    - logprobs      (n, #actions) float
    - state         tuple of hidden and cell states
    """
    if state is None:
      state = self.init_hidden(img_feat.shape[0])
    
    img_feat = self.cnn_fc_layer(img_feat)     # (n, fc_dim)
    emb_feat = self.phrase_embed(phrase_emb)   # (n, fc_dim)
    act_feat = self.action_embed(action_input) # (n, fc_dim)
    fuse_t = torch.mul(img_feat, emb_feat)     # (n, fc_dim)
    if self.use_action:
      fuse_t = torch.cat([fuse_t, act_feat], 1)  # (n, fc_dim * 2)
    # rnn
    output, state = self.core(fuse_t, state)  # (n, rnn_size)
    logprobs = self.act_gen(output)  # (n, #actions)
    return logprobs, state

