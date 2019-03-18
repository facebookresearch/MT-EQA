"""
Model definitions of VQA baseline models.
1) QA only
2) Abhishek's single-stream model
3) multi-stream model
"""
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

"""
QA LSTM Model
"""
class QaLstmModel(nn.Module):
  def __init__(self,
               vocab,
               rnn_wordvec_dim=64,
               rnn_dim=64,
               rnn_num_layers=2,
               rnn_dropout=0.5,
               fc_use_batchnorm=False,
               fc_dropout=0.5,
               fc_dims=(64, )):
    super(QaLstmModel, self).__init__()
    rnn_kwargs = {
        'token_to_idx': vocab['questionTokenToIdx'],
        'wordvec_dim': rnn_wordvec_dim,
        'rnn_dim': rnn_dim,
        'rnn_num_layers': rnn_num_layers,
        'rnn_dropout': rnn_dropout,
    }
    self.rnn = QuestionLstmEncoder(**rnn_kwargs)

    classifier_kwargs = {
        'input_dim': rnn_dim,
        'hidden_dims': fc_dims,
        'output_dim': len(vocab['answerTokenToIdx']),
        'use_batchnorm': fc_use_batchnorm,
        'dropout': fc_dropout,
        'add_sigmoid': 0
    }
    self.classifier = build_mlp(**classifier_kwargs)

  def forward(self, questions):
    q_feats = self.rnn(questions)
    scores = self.classifier(q_feats)
    return scores

"""
VqaLstmCnnAttentionModel
Given a question and a set of last-5-frames at key moments, we use question to attentionally pool the visual features.
Then fuse the language and visual features together to predict the answer.
"""
class VqaLstmCnnAttentionModel(nn.Module):
  def __init__(self, vocab, 
               image_feat_dim=64,
               question_wordvec_dim=64,
               question_hidden_dim=64,
               question_num_layers=2,
               question_dropout=0.5,
               fc_use_batchnorm=False,
               fc_dropout=0.5,
               fc_dims=(64, )):
    super(VqaLstmCnnAttentionModel, self).__init__()
    # visual encoder
    self.cnn_fc_layer = nn.Sequential(nn.Linear(32 * 10 * 10, 64), nn.ReLU(), nn.Dropout(p=0.5))
    # question encoder
    q_rnn_kwargs = {
      'token_to_idx': vocab['questionTokenToIdx'],
      'wordvec_dim': question_wordvec_dim,
      'rnn_dim': question_hidden_dim,
      'rnn_num_layers': question_num_layers,
      'rnn_dropout': question_dropout,
    }
    self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)
    # two translations
    self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
    self.ques_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
    # answer classifier
    classifier_kwargs = {
      'input_dim': 64,
      'hidden_dims': fc_dims,
      'output_dim': len(vocab['answerTokenToIdx']),
      'use_batchnorm': fc_use_batchnorm,
      'dropout': fc_dropout,
      'add_sigmoid': 0
    }
    self.classifier = build_mlp(**classifier_kwargs)
    self.attn = nn.Sequential(nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1))
  
  def forward(self, img_feats, questions):
    """
    Inputs:
    - img_feats: bs x T x 3200, where T = 3*5
    - questions: bs x L
    Outputs:
    - scores     : bs x #answers
    - attn_probs : bs x T
    """
    N, T, _ = img_feats.size()
    img_feats = self.cnn_fc_layer(img_feats).view(N*T, -1)  # (NT, 64)
    img_feats_tr = self.img_tr(img_feats)  # (NT, 64)

    ques_feats = self.q_rnn(questions)  # (N, hidden_size)
    ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1)  # (N, T, hidden_size)
    ques_feats_repl = ques_feats_repl.view(N*T, -1) 
    ques_feats_tr = self.ques_tr(ques_feats_repl)  # (NT, 64)

    ques_img_feats = torch.cat([img_feats_tr, ques_feats_tr], 1)  # (NT, 128)
    attn_feats = self.attn(ques_img_feats)  # (NT, 1)
    attn_probs = F.softmax(attn_feats.view(N, T), dim=1)  # (N, T)
    attn_probs2 = attn_probs.view(N, T, 1).repeat(1, 1, 64)  # (N, T, 64)
    attn_img_feats = torch.mul(attn_probs2, img_feats.view(N, T, 64))  # (N, T, 64)
    attn_img_feats = torch.sum(attn_img_feats, dim=1)  # (N, 64)

    mul_feats = torch.mul(ques_feats, attn_img_feats) # (N, 64)
    scores = self.classifier(mul_feats)  # (N, #answers)
    return scores, attn_probs

"""
Takes three streams of image data and use three rnns to encode questions, hopefully we
will get three attentional visual representations.
Then concatenate them together to predict answers.
This would keep input visual features' order.
"""
class VqaLstmCnnMultiStreamAttentionModel(nn.Module):
  def __init__(self, vocab, 
               image_feat_dim=64,
               question_wordvec_dim=64,
               question_hidden_dim=64,
               question_num_layers=2,
               question_dropout=0.5,
               fc_use_batchnorm=False,
               fc_dropout=0.5,
               fc_dims=(64, )):
    super(VqaLstmCnnMultiStreamAttentionModel, self).__init__()
    # visual encoder
    self.cnn_fc_layer = nn.Sequential(nn.Linear(32 * 10 * 10, 64), nn.ReLU(), nn.Dropout(p=0.5))
    # question encoder
    q_rnn_kwargs = {
      'token_to_idx': vocab['questionTokenToIdx'],
      'wordvec_dim': question_wordvec_dim,
      'rnn_dim': question_hidden_dim,
      'rnn_num_layers': question_num_layers,
      'rnn_dropout': question_dropout,
    }
    self.q_rnn1 = QuestionLstmEncoder(**q_rnn_kwargs)
    self.q_rnn2 = QuestionLstmEncoder(**q_rnn_kwargs)
    self.q_rnn3 = QuestionLstmEncoder(**q_rnn_kwargs)
    # two translations
    self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
    self.ques_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
    # answer classifier
    classifier_kwargs = {
      'input_dim': 64 * 3,
      'hidden_dims': fc_dims,
      'output_dim': len(vocab['answerTokenToIdx']),
      'use_batchnorm': fc_use_batchnorm,
      'dropout': fc_dropout,
      'add_sigmoid': 0
    }
    self.classifier = build_mlp(**classifier_kwargs)
    self.attn = nn.Sequential(nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1))
  
  def forward(self, img_feats1, img_feats2, img_feats3, question):
    """
    Inputs:
    - img_feats1 (n, T, 3200), where T = 5
    - img_feats2 (n, T, 3200), where T = 5
    - img_feats3 (n, T, 3200), where T = 5
    - question is of size (n, L)
    Outputs:
    - scores     : n x #answers
    - attn_probs : n x 3 x T
    """
    N, T, _ = img_feats1.size()
    mul_feats_list = []
    attn_probs_list = []
    for img_feats, q_rnn in zip([img_feats1, img_feats2, img_feats3], [self.q_rnn1, self.q_rnn2, self.q_rnn3]):

      img_feats = self.cnn_fc_layer(img_feats).view(N*T, -1)  # (NT, 64)
      img_feats_tr = self.img_tr(img_feats)  # (NT, 64)
      ques_feats = q_rnn(question)  # (N, hidden_size)
      ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1) # (N, T, hidden_size)
      ques_feats_repl = ques_feats_repl.view(N * T, -1)  # (NT, hidden_size)
      ques_feats_tr = self.ques_tr(ques_feats_repl)  # (NT, 64)

      ques_img_feats = torch.cat([ques_feats_tr, img_feats_tr], 1)  # (NT, 128)
      attn_feats = self.attn(ques_img_feats)  # (NT, 1)
      attn_probs = F.softmax(attn_feats.view(N, T), dim=1)  # (N, T)
      attn_probs2 = attn_probs.view(N, T, 1).repeat(1, 1, 64)  # (N, T, 64)
      attn_img_feats = torch.mul(attn_probs2, img_feats.view(N, T, 64)) # (N, T, 64)
      attn_img_feats = torch.sum(attn_img_feats, dim=1)  # (N, 64)
      mul_feats = torch.mul(ques_feats, attn_img_feats)  # (N, 64)
      mul_feats_list.append(mul_feats)
      attn_probs_list.append(attn_probs)
    
    # compute final scores
    mul_feats = torch.cat(mul_feats_list, 1)  # (n, 3 x 64)
    scores = self.classifier(mul_feats)  # (n, #answers)
    # attn_probs for 3 img_feats  (N, 3, 5)
    attn_probs = torch.cat([_.unsqueeze(1) for _ in attn_probs_list], 1)  # (n, 3, 5)
    return scores, attn_probs
    
