# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model definitions of task-driven VQA model.
"""
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
MLP layer: 
[nn.Linear(input_dim, hidden_dim) + nn.ReLU]s + nn.Linear(hidden_dim, output_dim) + nn.ReLU
"""
def build_mlp(input_dim, hidden_dims, output_dim, use_batchnorm=False,
              dropout=0, add_sigmoid=1):
  layers = []
  D = input_dim
  if dropout > 0:
    layers.append(nn.Dropout(p=dropout))
  if use_batchnorm:
    layers.append(nn.BatchNorm1d(input_dim))
  for dim in hidden_dims:
    layers.append(nn.Linear(D, dim))
    if use_batchnorm:
      layers.append(nn.BatchNorm1d(dim))
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
    layers.append(nn.ReLU(inplace=True))
    D = dim
  layers.append(nn.Linear(D, output_dim))

  if add_sigmoid == 1:
    layers.append(nn.Sigmoid())
  return nn.Sequential(*layers)

"""
Question Encoder using single-directional LSTM
We extract the last hidden state at <END> token as question feature.
"""
class QuestionLstmEncoder(nn.Module):
  def __init__(self, token_to_idx, wordvec_dim=64, rnn_dim=64, rnn_num_layers=2, rnn_dropout=0):
    super(QuestionLstmEncoder, self).__init__()
    self.token_to_idx = token_to_idx
    self.NULL = token_to_idx['<NULL>']
    self.START = token_to_idx['<START>']
    self.END = token_to_idx['<END>']
    
    self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
    self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers, dropout=rnn_dropout, batch_first=True)
    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.embed.weight.data.uniform_(-initrange, initrange)

  def forward(self, x):
    N, T = x.size()
    idx = torch.LongTensor(N).fill_(T - 1)
    # Find the last non-null element in each sequence
    x_cpu = x.data.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data).long()
    idx = Variable(idx, requires_grad=False)
    # Fetch last hidden
    hs, _ = self.rnn(self.embed(x))  # (batch, seq_len, hidden_size)
    idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))  # (batch, 1, hidden_size)
    H = hs.size(2)
    return hs.gather(1, idx).view(N, H)  # (batch, hidden_size)

class VqaLstmCnnTaskStreamAttentionModel(nn.Module):
  def __init__(self, vocab,
               image_feat_dim=64,
               question_wordvec_dim=64,
               question_hidden_dim=64,
               question_num_layers=2,
               question_dropout=0.5,
               fc_use_batchnorm=False,
               fc_dropout=0.5,
               fc_dims=(64, )):
    super(VqaLstmCnnTaskStreamAttentionModel, self).__init__()
    # visual encoder
    self.cnn_fc_layer = nn.Sequential(nn.Linear(32 * 10 * 10, 64), nn.ReLU(), nn.Dropout(p=0.5))
    # question encoder
    q_rnn_kwargs = 