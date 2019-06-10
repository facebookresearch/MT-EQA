# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .StackRNNCell import StackedRNN, StackedLSTM

"""
Localizer
"""
class Localizer(nn.Module):

  def __init__(self, opt):
    super(Localizer, self).__init__()
    # options
    self.rnn_type = opt.get('rnn_type', 'lstm')
    assert self.rnn_type in ['rnn', 'gru', 'lstm']
    self.rnn_size = opt.get('rnn_size', 256)
    self.num_layers = opt.get('num_layers', 1)
    self.rnn_dropout = opt.get('rnn_dropout', 0.2)  # rnn dropout
    self.rnn_fc_dim = opt.get('rnn_fc_dim', 64)
    self.seq_dropout = opt.get('seq_dropout', 0.)  # dropout for generator layer

    # img embedding
    self.cnn_fc_layer = nn.Sequential(nn.Linear(3200, self.rnn_fc_dim), nn.ReLU(), nn.Dropout(p=self.seq_dropout))

    # fusion layer
    self.phrase_embed = nn.Sequential(nn.Linear(300, self.rnn_fc_dim), nn.ReLU(), nn.Dropout(p=self.seq_dropout))

    # create core rnn
    if self.rnn_type in ['rnn', 'gru']:
      self.core = StackedRNN(self.num_layers, self.rnn_fc_dim, self.rnn_size, self.rnn_dropout, self.rnn_type)
    else:
      self.core = StackedLSTM(self.num_layers, self.rnn_fc_dim, self.rnn_size, self.rnn_dropout)

    # answer --> logprobs
    self.classifier = nn.Sequential(
                        nn.Linear(self.rnn_size, self.rnn_fc_dim), 
                        nn.ReLU(), 
                        nn.Dropout(p=self.seq_dropout),
                        nn.Linear(self.rnn_fc_dim, 2),
                        nn.LogSoftmax(dim=1)
                        )

  def init_hidden(self, batch_size):
    """first hidden vector(s)"""
    weight = next(self.parameters()).data
    if self.rnn_type == 'lstm':
      return (weight.new(self.num_layers, batch_size, self.rnn_size).zero_(),
              weight.new(self.num_layers, batch_size, self.rnn_size).zero_())
    else:
      return weight.new(self.num_layers, batch_size, self.rnn_size).zero_()

  def forward(self, img_feats, phrase_emb, masks, select_ixs):
    """
    Inputs:
    - img_feats  (n, L, 3200)
    - phrase_emb (n, 300)
    - masks      (n, L)
    Outputs:
    - logprobs       (n, L, 2)
    - output_feats   (n, L, rnn_size)
    - selected_feats (n, rnn_size)
    """
    assert img_feats.shape[0] == phrase_emb.shape[0]
    batch_size, L = img_feats.shape[0], img_feats.shape[1]

    # initialize
    emb_feats = self.phrase_embed(phrase_emb)  # (n, fc_dim)
    state = self.init_hidden(batch_size)
    logprobs = []
    output_feats = []
    for t in range(L):
      # if masks[:, t:].sum() == 0:  # break if no effective data since then
      #   break
      img_t = img_feats[:, t, :]  # (n, 3200)
      img_t = self.cnn_fc_layer(img_t)  # (n, fc_dim)
      emb_t = emb_feats  # (n, fc_dim)
      fuse_t = torch.mul(img_t, emb_t)  # (n, fc_dim)
      # rnn
      output_t, state = self.core(fuse_t, state) # (n, rnn_size)
      # generate
      logprobs_t = self.classifier(output_t)     # (n, 2)
      logprobs.append(logprobs_t.unsqueeze(1))   # (n, 1, 2)
      output_feats.append(output_t.unsqueeze(1)) # (n, 1, rnn_size)
    # output
    logprobs = torch.cat(logprobs, 1)  # (n, effective_length, 2)
    output_feats = torch.cat(output_feats, 1)  # (n, effective_length, rnn_size)
    # select
    selected_feats = self.select(output_feats, select_ixs) # (n, rnn_size)
    # return
    return logprobs, output_feats, selected_feats

  def forward_test(self, img_feats, phrase_emb):
    """
    Inputs:
    - img_feats  (n, L, 3200)
    - phrase_emb (n, 300)
    Outputs:
    - logprobs     (n, L, 2)
    - output_feats (n, L, rnn_size)
    - state        tuple of hidden and cell states
    """
    assert img_feats.shape[0] == phrase_emb.shape[0]
    batch_size, L = img_feats.shape[0], img_feats.shape[1]

    # initialize
    emb_feats = self.phrase_embed(phrase_emb)  # (n, fc_dim)
    state = self.init_hidden(batch_size)
    logprobs = []
    output_feats = []
    for t in range(L):
      img_t = img_feats[:, t, :]  # (n, 3200)
      img_t = self.cnn_fc_layer(img_t)  # (n, fc_dim)
      emb_t = emb_feats  # (n, fc_dim)
      fuse_t = torch.mul(img_t, emb_t)  # (n, fc_dim)
      # rnn
      output_t, state = self.core(fuse_t, state)  # (n, rnn_size)
      # generate
      logprobs_t = self.classifier(output_t)     # (n, 2)
      logprobs.append(logprobs_t.unsqueeze(1))   # (n, 1, 2)
      output_feats.append(output_t.unsqueeze(1)) # (n, 1, rnn_size)
    # output
    logprobs = torch.cat(logprobs, 1)  # (n, L, 2)
    output_feats = torch.cat(output_feats, 1)  # (n, L, rnn_size)
    return logprobs, output_feats, state

  def forward_step(self, img_feat, phrase_emb, state=None):
    """
    Inputs:
    - img_feat      (n, 3200) float
    - phrase_emb    (n, 300) float
    - state         tuple of hidden and cell states
    Output:
    - logprobs      (n, #actions) float
    - rnn_feats     (n, rnn_size) float
    - state         tuple of hidden and cell states
    """
    # initialize
    if state is None:
      state = self.init_hidden(img_feat.shape[0])
    img_feat = self.cnn_fc_layer(img_feat)   # (n, fc_dim)
    emb_feat = self.phrase_embed(phrase_emb)  # (n, fc_dim)
    fuse_feat = torch.mul(img_feat, emb_feat)  # (n, fc_dim)
    # forwar core 
    output_feats, state = self.core(fuse_feat, state)  # (n, rnn_size), state
    # generate
    logprobs = self.classifier(output_feats)   # (n, 2)
    # output
    return logprobs, output_feats, state

  def select(self, feats, select_ixs):
    """
    Inputs:
    - feats      (n, L, V) float
    - select_ixs (n, ) long
    Output:
    - selected_feats (n, V) float
    We select one feat per row.
    """
    n, L, V = feats.shape[0], feats.shape[1], feats.shape[2]
    assert n == select_ixs.shape[0]
    select_ixs = select_ixs.view(n, 1).expand(n, V).unsqueeze(1)  # (n, 1, V)
    selected_feats = feats.gather(1, select_ixs)  # (n, 1, V)
    selected_feats = selected_feats.view(n, V)  # (n, V)
    return selected_feats

