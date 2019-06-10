# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model definitions for baseline localized-VQA, which is VQA without localization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
Question encoder
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

    hs, _ = self.rnn(self.embed(x))
    idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
    H = hs.size(2)
    return hs.gather(1, idx).view(N, H)

"""
Use a single LSTM to attend the video sequence, and get an attentional visual representation.
Then fuse it with question again for answer prediction.
"""
class SingleAttentionVQAModel(nn.Module):
  def __init__(self, wtoi, image_feat_dim=64, question_wordvec_dim=64,
               question_hidden_dim=64, question_num_layers=2, question_dropout=0.5,
               fc_use_batchnorm=False, fc_dropout=0.5, fc_dims=(64, )):
    super(SingleAttentionVQAModel, self).__init__()
    # cnn fc layer
    self.cnn_fc_layer = nn.Sequential(nn.Linear(32*10*10, 64), nn.ReLU(), nn.Dropout(p=0.5))
    # three rnn encoder
    q_rnn_kwargs = {
      'token_to_idx': wtoi['question'],
      'wordvec_dim': question_wordvec_dim,
      'rnn_dim': question_hidden_dim,
      'rnn_num_layers': question_num_layers,
      'rnn_dropout': question_dropout,
    }
    self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)
    # more mlps for answer prediction
    self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
    self.ques_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
    self.att = nn.Sequential(nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1))
    # classifier
    classifier_kwargs = {
      'input_dim': 64,
      'hidden_dims': fc_dims,
      'output_dim': len(wtoi['answer']),
      'use_batchnorm': fc_use_batchnorm,
      'dropout': fc_dropout,
      'add_sigmoid': 0
    }
    self.classifier = build_mlp(**classifier_kwargs)

  def forward(self, img_feats, questions):
    """
    Inputs:
    - img_feats (n, Lt, 32 x 10 x 10)
    - questions (n, Lq)
    Outputs:
    - answers   (n, #answers) 
    - att_probs (n, Lt)
    """
    # n x (Lt) x 3 x 224 x 224 --cnn--> n x Lt x 3200
    N, T, _ = img_feats.size()
    img_feats = self.cnn_fc_layer(img_feats).view(N*T, -1)  # (n x Lt, 64)
    img_feats_tr = self.img_tr(img_feats)  # (n x Lt, 64)

    ques_feats = self.q_rnn(questions)  # (n, h)
    ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1) # (n, Lt, h)
    ques_feats_repl = ques_feats_repl.view(N*T, -1)  # (n x Lt, h)
    ques_feats_tr = self.ques_tr(ques_feats_repl)  # (n x Lt, 64)

    ques_imgs_feats = torch.cat([ques_feats_tr, img_feats_tr], 1)  
    att_feats = self.att(ques_imgs_feats)  # (nxLt, 1)
    att_probs = F.softmax(att_feats.view(N, T), dim=1)  # (n, Lt)
    att_probs2 = att_probs.view(N, T, 1).repeat(1, 1, 64)  # (n, Lt, 64)
    att_img_feats = torch.mul(att_probs2, img_feats.view(N, T, 64))
    att_img_feats = torch.sum(att_img_feats, dim=1)

    mul_feats = torch.mul(ques_feats, att_img_feats)
    scores = self.classifier(mul_feats)
    return scores, att_probs