# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .navigator import Navigator
from .localizer import Localizer
from .vqa_model import ModularAttributeVQA

class ModelFactory(nn.Module):
  def __init__(self, opt):
    super(ModelFactory, self).__init__()

    # set up navigator
    self.object_navigator = Navigator(opt)
    self.room_navigator = Navigator(opt)

    # set up localizer
    self.object_localizer = Localizer(opt)
    self.room_localizer = Localizer(opt)

    # set up vqa
    self.vqa = ModularAttributeVQA(ego_feat_dim = 3200,
                                   rnn_feat_dim = opt['rnn_size'],
                                   cube_feat_dim = 3200,
                                   fc_dim = opt['qn_fc_dim'],
                                   fc_dropout = opt['qn_fc_dropout'],
                                   num_answers = len(opt['atoi']))

  def forward_step(self, ego_feats, phrase_emb, nav_type, nav_action, nav_state, loc_state):
    """
    Inputs:
    - ego_feats   (n, 3200)
    - phrase_emb  (n, 300)
    - nav_type   
    - nav_action  (n, ) int
    - nav_state   (n, rnn_size) float
    - loc_state   (n, rnn_size) float
    Outputs:
    - nav_logprobs  (n, 4) float {0: forward, 1: left, 2: right, 3: stop}
    - nav_state     (n, rnn_size) float
    - loc_logprobs  (n, 4) float {0: nothing, 1: store}
    - loc_rnn_feats (n, rnn_size) float
    - loc_state     (n, rnn_size) float
    """
    navigator = self.object_navigator if nav_type == 'object' else self.room_navigator
    localizer = self.object_localizer if nav_type == 'object' else self.room_localizer
    nav_logprobs, nav_state = navigator.forward_step(ego_feats, phrase_emb, nav_action, nav_state)
    loc_logprobs, loc_rnn_feats, loc_state = localizer.forward_step(ego_feats, phrase_emb, loc_state)
    return nav_logprobs, nav_state, loc_logprobs, loc_rnn_feats, loc_state

  def compute_states(self, nav_type, ego_feats, phrase_embs, action_inputs):
    """
    Forward model the pre-path
    Inputs:
    - nav_type      : room or object
    - ego_feats     : (n, l, 3200)
    - phrase_embs   : (n, 300)
    - action_inputs : (n, l)
    Outputs:
    - nav_state     : (1, rnn_size)
    - loc_state     : (1, rnn_size)
    """
    navigator = self.object_navigator if nav_type == 'object' else self.room_navigator
    localizer = self.object_localizer if nav_type == 'object' else self.room_localizer
    _, _, _, nav_state = navigator(ego_feats, phrase_embs, action_inputs)
    _, _, loc_state = localizer.forward_test(ego_feats, phrase_embs)
    return nav_state, loc_state

  def forward(self, object_ego_feats, object_phrase_embs, object_masks, object_select_ixs, object_action_inputs,
              room_ego_feats, room_phrase_embs, room_masks, room_select_ixs, room_cube_feats, room_action_inputs,
              attrs):
    """
    Inputs:
    - object_ego_feats     (n, 3, L, 3200) float
    - object_phrase_embs   (n, 3, 300) float
    - object_masks         (n, 3, L) float
    - object_select_ixs    (n, 3) long
    - object_action_inputs (n, 3, L) int
    -----------------
    - room_ego_feats       (n, 2, L, 3200) float
    - room_masks           (n, 2, L) float
    - room_select_ixs      (n, 2) long (actually useless here, we use room_cube_feats instead)
    - room_cube_feats      (n, 2, 4, 3200) float
    - room_action_inputs   (n, 2, L) int
    -----------------
    - attrs                list of n question attributes
    -----------------
    Outputs:
    - room_action_logprobs   (nx2, L, #nav_acts)
    - object_action_logprobs (nx3, L, #nav_acts)
    - room_loc_logprobs      (nx2, L, #store_acts), #store_acts = 2
    - object_loc_logprobs    (nx3, L, #store_acts)
    - scores                 (n, #answers)
    """
    # batch process everything
    n = object_ego_feats.shape[0]  # number of questions
    assert room_ego_feats.shape[0] == room_cube_feats.shape[0] == n

    object_ego_feats = object_ego_feats.view(n*3, -1, 3200)  # (nx3, L, 3200)
    object_phrase_embs = object_phrase_embs.view(n*3, 300)   # (nx3, 300)
    object_masks = object_masks.view(n*3, -1)       # (nx3, L)
    object_select_ixs = object_select_ixs.view(-1)  # (nx3, )
    object_action_inputs = object_action_inputs.view(n*3, -1)  # (nx3, L)

    room_ego_feats = room_ego_feats.view(n*2, -1, 3200) # (nx2, L, 3200)
    room_phrase_embs = room_phrase_embs.view(n*2, 300)  # (nx2, 300)
    room_masks = room_masks.view(n*2, -1)       # (nx2, L)
    room_select_ixs = room_select_ixs.view(-1)  # (nx2, )
    room_action_inputs = room_action_inputs.view(n*2, -1)  # (nx2, L)

    # forward navigator - logprobs (nk, l, 5)
    object_action_logprobs, _, _, _ = self.object_navigator(object_ego_feats, object_phrase_embs, object_action_inputs)
    room_action_logprobs, _, _, _ = self.room_navigator(room_ego_feats, room_phrase_embs, room_action_inputs)

    # forward localizer
    # - logprobs (nk, l, 2)
    # - output_feats (nk, l, rnn_size)
    # - selected_feats (nk, rnn_size)
    selected_ego_feats = self.object_localizer.select(object_ego_feats, object_select_ixs) # (nx3, 3200)
    object_loc_logprobs, _, selected_rnn_feats = self.object_localizer(object_ego_feats, object_phrase_embs, object_masks, object_select_ixs)
    room_loc_logprobs, _, _ = self.room_localizer(room_ego_feats, room_phrase_embs, room_masks, room_select_ixs)

    # prepare list of feats
    selected_ego_feats = selected_ego_feats.view(n, 3, -1)  # (n, 3, 3200)
    selected_rnn_feats = selected_rnn_feats.view(n, 3, -1)  # (n, 3, rnn_size)
    img_feats_list = []
    for i, attr in enumerate(attrs):
      if 'room_size' in attr:
        img_feats_list.append(room_cube_feats[i]) # room_cube_feats (2, 4, 3200)
      elif 'object_dist' in attr:
        img_feats = torch.cat([selected_ego_feats[i], selected_rnn_feats[i]], 1)  # (3, 3200+rnn_size)
        img_feats_list.append(img_feats) # [obj_feat1, obj_feat2, obj_feat3] (3, 3200+rnn_size)
      else:
        img_feats_list.append(selected_rnn_feats[i, :2, :]) # [obj_feat1, obj_feat2] (2, rnn_size)

    # forward vqa module
    scores = self.vqa(img_feats_list, attrs)

    # return
    return room_action_logprobs, object_action_logprobs, room_loc_logprobs, object_loc_logprobs, scores
  
  def gd_path_key_to_answer(self, ego_feats, cube_feats, tgt_phrase_embs, tgt_key_ixs, attr):
    """
    Inputs:
    - ego_feats       (L, 3200) float 
    - cube_feats      (L, 4, 3200) float
    - tgt_phrase_embs (#targets, 300) float, for to-compare targets
    - tgt_key_ixs     (#targets, ) long, indices for to-compare targets
    - attr
    Outputs:
    - ans_scores      (#answers, ) float, after softmax
    """
    n = tgt_phrase_embs.shape[0] # number of to-compare-targets
    L = ego_feats.shape[0]       # path length
    if 'room_size' in attr:
      assert tgt_key_ixs.shape[0] == 2, 'only two rooms are considered for room_compare'
      img_feats = [cube_feats[key_ix] for key_ix in tgt_key_ixs.tolist()]
      img_feats = torch.cat(img_feats, 0)  # (#targets, 4, 3200)
    else:
      expanded_ego_feats = ego_feats.unsqueeze(0).expand(n, L, 3200)  # (n, L, 3200)
      _, rnn_feats, _ = self.object_localizer.forward_test(expanded_ego_feats, tgt_phrase_embs)  # (n, L, 2), (n, L, rnn_size)
      selected_rnn_feats = self.object_localizer.select(rnn_feats, tgt_key_ixs)  # (n, rnn_size)
      if 'dist' in attr:
        selected_ego_feats = ego_feats[tgt_key_ixs]  # (n, 3200)
        img_feats = torch.cat([selected_ego_feats, selected_rnn_feats], 1)  # (n, 3200+rnn_size), n=3
      else:
        # color, size
        img_feats = selected_rnn_feats

    # forward vqa
    scores = self.vqa([img_feats], [attr])  # (1, #answers)
    scores = F.softmax(scores, 1)  # (1, #answers)
    # return
    return scores.view(-1)

  def gd_path_to_sample_answer(self, ego_feats, cube_feats, nav_phrase_embs, nav_types, attr):
    """
    Inputs:
    - ego_feats           (L, 3200) float 
    - cube_feats          (L, 4, 3200) float
    - nav_phrase_embs     (#nav, 300) float
    - nav_types           #nav object or room
    - attr                
    Outputs:
    - loc_probs           (#nav, L), [0, 1) after softmax
    - sample_ixs          (#nav, )
    - ans_scors           (#answers, ) after softmax
    """
    # compute loc_probs (#nav, L)
    loc_probs = []       # list of #nav logprobs (1, L, 2)
    rnn_feats_list = []  # list of #obj (1, L, rnn_size)
    for i, t in enumerate(nav_types):
      if t == 'object':
        # object_logprobs (1, L, 2), object_feats (1, L, rnn_size)
        loc_object_logprobs, rnn_feats, _ = self.object_localizer.forward_test(ego_feats.unsqueeze(0), nav_phrase_embs[i].unsqueeze(0))
        loc_probs.append(loc_object_logprobs)
        rnn_feats_list.append(rnn_feats)
      else:
        # room_logprobs (1, L, 2)
        loc_room_logprobs, _, _ = self.room_localizer.forward_test(ego_feats.unsqueeze(0), nav_phrase_embs[i].unsqueeze(0))
        loc_probs.append(loc_room_logprobs)

    # sample
    loc_probs = torch.cat(loc_probs, 0)      # (#nav, L, 2)
    loc_probs = torch.exp(loc_probs[:,:,1])  # (#nav, L) for predicting "Storing" action
    sample_ixs = []
    prev = -1
    L = ego_feats.shape[0] # path_len
    n = len(nav_types)     # navs
    for i in range(n):
      sample_ix = L-1
      for k, sc in enumerate(loc_probs[i]): # (L)
        if sc >= 0.5 and k > prev:
          sample_ix = k
          break
      sample_ixs.append(sample_ix)
      prev = sample_ix 

    # prepare img_feats
    sample_ixs = torch.LongTensor(sample_ixs).cuda()
    if 'object' in attr:
      rnn_feats = torch.cat(rnn_feats_list, 0)  # (#obj, L, rnn_size)
      ixs = [sample_ixs[i] for i, t in enumerate(nav_types) if t == 'object']
      assert len(ixs) == rnn_feats.shape[0]
      sampled_rnn_feats = self.object_localizer.select(rnn_feats, torch.LongTensor(ixs).cuda())  # (#obj, rnn_size)
      if 'dist' in attr:
        sampled_ego_feats = ego_feats[torch.LongTensor(ixs).cuda()]  # (#obj, 3200)
        img_feats = torch.cat([sampled_ego_feats, sampled_rnn_feats], 1)  # (#obj, 3200+rnn_size)
      else:
        img_feats = sampled_rnn_feats  # (#obj, rnn_size)
    else:
      assert sample_ixs.shape[0] == 2, 'sample_ixs is of shape %s' % sample_ixs.shape
      img_feats = cube_feats.index_select(0, sample_ixs)  # (2, 4, 3200)

    # forward vqa
    scores = self.vqa([img_feats], [attr])  # (1, #answers)
    scores = F.softmax(scores, 1)  # (1, #answers)

    # return
    return loc_probs, sample_ixs, scores.view(-1)

