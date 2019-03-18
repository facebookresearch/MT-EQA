from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

"""
Stacked RNN/GRU
"""
class StackedRNN(nn.Module):
  def __init__(self, num_layers, input_size, rnn_size, dropout, rnn_type='rnn'):
    super(StackedRNN, self).__init__()
    assert rnn_type in ['rnn', 'gru']
    self.dropout = nn.Dropout(dropout)
    self.num_layers = num_layers

    for i in range(num_layers):
      if rnn_type == 'rnn':
        layer = nn.RNNCell(input_size, rnn_size)
      if rnn_type == 'gru':
        layer = nn.GRUCell(input_size, rnn_size)
      self.add_module('layer_%d' % i, layer)
      input_size = rnn_size

  def forward(self, input, hidden):
    """
    inputs:
    - input: (batch_size, input_size)
    - h0   : (layer_size, batch_size, rnn_size)
    output (after one step):
    - hout : (batch_size, rnn_size)
    - h1   : (layer_size, batch_size, rnn_size)
    """
    h_0 = hidden
    h_1 = []
    for i in range(self.num_layers):
      layer = getattr(self, 'layer_%d' % i)
      h_1_i = layer(input, h_0[i])
      input = h_1_i
      if i != self.num_layers:
        input = self.dropout(input)  # dropout between layers, but not on output!
      h_1 += [h_1_i]  # dropout is not on output!
    h_1 = torch.stack(h_1)
    return input, h_1

"""
Stacked LSTM
"""
class StackedLSTM(nn.Module):
  def __init__(self, num_layers, input_size, rnn_size, dropout):
    super(StackedLSTM, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.num_layers = num_layers

    for i in range(num_layers):
      layer = nn.LSTMCell(input_size, rnn_size)
      self.add_module('layer_%d' % i, layer)
      input_size = rnn_size

  def forward(self, input, hidden):
    """
    inputs:
    - input   : (batch_size, input_size)
    - hidden  : (h0, c0), each is (layer_size, batch_size, rnn_size)
    output (after one step):
    - hout    : (batch_size, rnn_size)
    - (h1, c1): each is (layer_size, batch_size, rnn_size)
    """
    h_0, c_0 = hidden
    h_1, c_1 = [], []
    for i in range(self.num_layers):
      layer = getattr(self, 'layer_%d' % i)
      h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
      input = h_1_i
      if i != self.num_layers:
        input = self.dropout(input)  # dropout between layers, but not on output!
      h_1 += [h_1_i]  # dropout is not on output!
      c_1 += [c_1_i]  # dropout is not on output!
    h_1 = torch.stack(h_1)
    c_1 = torch.stack(c_1)
    return input, (h_1, c_1)