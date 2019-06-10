# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import json
import h5py
import numpy as np
import random

import torch
from torch.utils.data import Dataset

"""
Navigator-Controller-VQA Imitaion Learning Dataset
We load
1) cache/prepro/data.json
- questions: [{h5_id, id, question, answer, split, path_name, key_ixs_set, num_paths, bbox, program, 
               room_labels, room_inroomDists}]
  where bbox = [{id, phrase, room_id, room_phrase, box}]
- wtoi: question vocab
- atoi: answer vocab
- ctoi: color vocab
- wtov: word2vec
We use path_name to get imgs_h5 {num_paths, orderedK, ego_rgbk, ego_semk, cube_rbbk, key_ixsk, 
                                 positionsk, actions}
and feats_h5.
2) cache/prepro/data.h5
- encoded_questions  (N, L)
- encoded_answers (N, )
"""
class NavLocVqaImDataset(Dataset):

  def __init__(self, data_json, data_h5, 
               path_feats_dir, path_images_dir,
               split, 
               object_seq_length=50, 
               room_seq_length=100, 
               requires_imgs=False, 
               question_types=['all']):

    print('Loading data.json:', data_json)
    self.infos = json.load(open(data_json))
    self.wtoi = self.infos['wtoi']  # question vocab
    self.atoi = self.infos['atoi']  # answer vocab
    self.ctoi = self.infos['ctoi']  # color vocab
    self.wtov = self.infos['wtov']  # word2vec
    self.itow = {i: w for w, i in self.wtoi.items()}
    self.itoa = {i: a for a, i in self.atoi.items()}
    self.itoc = {i: c for c, i in self.ctoi.items()}
    print('%s question vocab, %s answer vocab, %s color vocab, %s word2vec loaded.' % \
      (len(self.wtoi), len(self.atoi), len(self.ctoi), len(self.wtov)))

    if question_types == ['all']:
      self.questions = [qn for qn in self.infos['questions'] if qn['split'] == split]
    else:
      self.questions = [qn for qn in self.infos['questions'] if qn['split'] == split and qn['type'] in question_types]
    print('%s questions loaded for type%s under split[%s].' % (len(self.questions), question_types, split))

    # construct mapping
    self.Questions = {qn['id']: qn for qn in self.questions}
    self.ids = [qn['id'] for qn in self.questions]

    # actions
    self.actions = ['forward', 'left', 'right', 'dummy']
    self.act_to_ix = {'forward': 0, 'left': 1, 'right': 2, 'stop': 3, 'dummy': 4}
    self.ix_to_act = {i: a for a, i in self.act_to_ix.items()}

    # more info
    self.split = split
    self.path_feats_dir = path_feats_dir
    self.path_images_dir = path_images_dir
    self.requires_imgs = requires_imgs
    self.pre_size = 5  # hard code it
    self.object_seq_length = object_seq_length
    self.room_seq_length = room_seq_length

    # load data.h5
    encoded = h5py.File(data_h5, 'r')
    self.encoded_questions = encoded['encoded_questions']
    self.encoded_answers = encoded['encoded_answers']
    assert self.encoded_questions.shape[0] == self.encoded_answers.shape[0]
    print('max_length of encoded_questions is', self.encoded_questions.shape[1])
    print('[%s] data prepared, where there are %s questions.' % (split, len(self.questions)))

  def question_to_attribute(self, qn):
    """
    attrs are [object_color_equal, object_size_bigger/smaller, object_dist_farther/closer, room_size_bigger/smaller]
    room_attrs are [inroom, xroom]
    """
    attr = ''
    if 'object_color' in qn['type']:
      attr = 'object_color_equal'
    elif 'object_size' in qn['type']:
      compare_type = 'bigger' if 'bigger' in qn['question'] else 'smaller'
      attr = 'object_size_%s' % compare_type
    elif 'room_size' in qn['type']:
      compare_type = 'bigger' if 'bigger' in qn['question'] else 'smaller'
      attr = 'room_size_%s' % compare_type
    elif 'object_dist' in qn['type']:
      compare_type = 'farther' if 'farther' in qn['question'] else 'closer'
      attr = 'object_dist_%s' % compare_type
    else:
      raise NotImplementedError
    room_attr = 'inroom' if 'inroom' in qn['type'] else 'xroom'
    return attr, room_attr

  def getTestData(self, qid, path_ix=0, requires_imgs=False):
    """
    Returned data contains:
    - qid
    - question, answer
    - nav_pgs
    - type
    - attr, room_attr
    - qe                  : (Lq, ) int64
    - ae                  : (1, ) int64
    - Boxes               : tgt_id -> box
    - path_ix         
    - path_len
    ------------
    - ego_feats           : (L, 3200) float32
    - cube_feats          : (L, 4, 3200) float32
    - tgt_ids             : target_ids (being asked to compare)
    - tgt_phrase_embs     : (#targets, 300) float32
    - tgt_phrases
    - tgt_key_ixs         : #targets ixs
    ------------
    - room_ids          
    - room_phrases      
    - room_phrase_embs    : (#targets, 300) float32
    - room_to_inroomDists : room_id --> inroomDists, each is of length L
    - room_to_labels      : room_id --> pre-defined label, each is of length L
    ------------
    - nav_ids             : list of #nav object_ids/room_ids
    - nav_phrases         : list of #nav phrases (object + room)
    - nav_phrase_embs     : (#nav, 300) float32
    - nav_types           : ['room', 'object', ...]
    - ego_sems            : (L, 224, 224, 3) uint8
    ------------
    - imgs                : (L, 224, 224, 3) uint8 if necessary
    """
    qn = self.Questions[qid]
    qid = qn['id']
    nav_pgs = [pg for pg in qn['program'] if 'nav' in pg['function']]

    # encode question and answer
    qe = self.encoded_questions[qn['h5_id']]
    ae = self.encoded_answers[qn['h5_id']]
    attr, room_attr = self.question_to_attribute(qn)
    Boxes = {box['id']: box for box in qn['bbox']}  # tgt_id -> box

    # fetch feats
    path_feats_h5 = h5py.File(osp.join(self.path_feats_dir, qn['path_name']+'.h5'), 'r')
    ego_feats = path_feats_h5['ego_rgb%s' % path_ix][...].reshape(-1, 3200)  # (L, 3200)
    cube_feats = path_feats_h5['cube_rgb%s' % path_ix][...].reshape(-1, 4, 3200)  # (L, 4, 3200)
    path_len = ego_feats.shape[0]
    # --
    path_imgs_h5 = h5py.File(osp.join(self.path_images_dir, qn['path_name']+'.h5'), 'r')
    ego_sems = path_imgs_h5['ego_sem%s' % path_ix][...]  # (L, 224, 224, 3)
    if requires_imgs:
      ego_imgs = path_imgs_h5['ego_rgb%s' % path_ix][...] # (L, 224, 224, 3)

    # nav_phrases, nav_phrase_embs
    nav_ids = [pg['id'][0] for pg in qn['program'] if 'nav' in pg['function']]
    nav_types = [pg['function'][4:] for pg in qn['program'] if 'nav' in pg['function']]
    nav_phrases = [pg['value_inputs'][0] for pg in qn['program'] if 'nav' in pg['function']]
    nav_phrase_embs = []
    for phrase in nav_phrases:
      nav_phrase_embs.append(np.array([self.wtov[wd] for wd in phrase.split()]).mean(0).astype(np.float32))  # (300, )
    nav_phrase_embs = np.array(nav_phrase_embs)  # (#targets, 300)

    # key_ixs
    raw_key_ixs = qn['key_ixs_set'][path_ix]
    tgt_key_ixs = self.get_target_key_ixs(raw_key_ixs, nav_pgs, qn)

    # tgt_ids: object_ids for object questions, room_ids for room questions
    tgt_ids = []
    if 'object' in attr:
      tgt_ids = [_id for _id, _type in zip(nav_ids, nav_types) if _type == 'object']
    elif 'room' in attr:
      tgt_ids = [_id for _id, _type in zip(nav_ids, nav_types) if _type == 'room']
    else:
      raise NotImplementedError
    assert len(tgt_ids) == len(tgt_key_ixs)

    # tgt_phrases, tgt_phrase_embs
    tgt_phrases = []
    tgt_phrase_embs = []
    for i, box in enumerate(qn['bbox']):
      assert tgt_ids[i] == box['id']
      tgt_phrase_embs.append(np.array([self.wtov[wd] for wd in box['phrase'].split()]).mean(0).astype(np.float32))  # (300,)
      tgt_phrases.append(box['phrase'])
    tgt_phrase_embs = np.array(tgt_phrase_embs)  # (#targets, 300)

    # room -> inroomDists, room -> labels, each is of path_len
    room_ids = [nav_id for nav_id, nav_type in zip(nav_ids, nav_types) if nav_type == 'room']
    room_phrases = [nav_phrase for nav_phrase, nav_type in zip(nav_phrases, nav_types) if nav_type == 'room']
    room_to_inroomDists = {room_id: qn['room_inroomDists'][room_id][path_ix] for room_id in room_ids} # each is (path_len, )
    room_to_labels = {room_id: qn['room_labels'][room_id][path_ix] for room_id in room_ids}  # each is (path_len, ) {0, 1}

    # return
    data = {}
    data['qid'] = qid
    data['question'] = qn['question']
    data['answer'] = qn['answer']
    data['nav_pgs'] = nav_pgs
    data['type'] = qn['type']
    data['attr'] = attr
    data['room_attr'] = room_attr
    data['qe'] = qe
    data['ae'] = ae
    data['Boxes'] = Boxes
    data['path_ix'] = path_ix
    data['path_len'] = path_len
    data['ego_feats'] = ego_feats
    data['tgt_key_ixs'] = tgt_key_ixs
    data['tgt_ids'] = tgt_ids
    data['cube_feats'] = cube_feats
    data['room_ids'] = room_ids
    data['room_phrases'] = room_phrases
    data['tgt_phrases'] = tgt_phrases
    data['tgt_phrase_embs'] = tgt_phrase_embs
    data['room_to_inroomDists'] = room_to_inroomDists
    data['room_to_labels'] = room_to_labels
    data['nav_ids'] = nav_ids
    data['nav_phrases'] = nav_phrases
    data['nav_phrase_embs'] = nav_phrase_embs
    data['nav_types'] = nav_types
    data['ego_sems'] = ego_sems
    if requires_imgs:
      data['imgs'] = ego_imgs
    return data

  def __getitem__(self, index):
    """
    Returned data contains
    - qid
    - question, answer
    - attr, room_attr
    - qe 
    - ae
    ------------
    - object_ego_feats      : (3, Lo, 3200) float32
    - object_phrases        : list of target names
    - object_phrase_embs    : (3, 300) float32
    - object_key_ixs        : (3, ) int64
    - object_masks          : (3, Lo) float32
    - object_labels         : (3, Lo) int64
    - object_action_inputs  : (3, Lo) int64
    - object_action_outputs : (3, Lo) int64
    - object_action_masks   : (3, Lo) float32
    ------------
    - room_ids    
    - room_ego_feats        : (2, Lr, 3200) float32
    - room_phrases          : list of room names
    - room_phrase_embs      : (2, 300) float32
    - room_cube_feats       : (2, 4, 3200) float32, randomly selected according to room_labels
    - room_key_ixs          : (2, ) int64, based on key_ixs
    - room_select_ixs       : (2, ) int64 selected from room_labels
    - room_masks            : (2, Lr) float32
    - room_labels           : (2, Lr) int64
    - room_action_inputs    : (2, Lr) float32
    - room_action_outputs   : (2, Lr) int64
    - room_action_masks     : (2, Lr) float32
    ------------
    - object_ego_imgs       : (3, Lo, 224, 224, 3) uint8 if necessary
    - room_ego_imgs         : (2, Lr, 224, 224, 3) uint8 if necessary
    """
    qn = self.questions[index]
    qid = qn['id']
    qe = self.encoded_questions[qn['h5_id']]
    ae = self.encoded_answers[qn['h5_id']]
    attr, room_attr = self.question_to_attribute(qn)
    nav_pgs = [pg for pg in qn['program'] if 'nav' in pg['function']]
    
    # choose path
    pix = random.choice(range(qn['num_paths'])) if self.split == 'train' else 0
    path_feats_h5 = h5py.File(osp.join(self.path_feats_dir, qn['path_name']+'.h5'), 'r')
    raw_ego_feats = path_feats_h5['ego_rgb%s' % pix]    # (raw_path_len, 32, 10, 10)
    raw_cube_feats = path_feats_h5['cube_rgb%s' % pix]  # (raw_path_len, 4, 32, 10, 10)
    raw_path_len = raw_ego_feats.shape[0]
    raw_actions = qn['path_actions'][pix]  # list of path_len actions {0:'forward', 1:'left', 2:'right', 3:'stop'}
    if self.requires_imgs:
      path_images_h5 = h5py.File(osp.join(self.path_images_dir, qn['path_name']+'.h5'), 'r')
      raw_ego_imgs = path_images_h5['ego_rgb%s' % pix]  # (n, 224, 224, 3)
      raw_cube_imgs = path_images_h5['cube_rgb%s' % pix]# (n, 4, 224, 224, 3) 
      object_ego_imgs = np.zeros((3, self.object_seq_length, 224, 224, 3), dtype=np.uint8)
      room_ego_imgs = np.zeros((2, self.room_seq_length, 224, 224, 3), dtype=np.uint8)
      room_selected_cube_imgs = np.zeros((2, 4, 224, 224, 3), dtype=np.uint8)

    # key_ixs
    raw_key_ixs = qn['key_ixs_set'][pix]
    assert len(raw_key_ixs) == len(nav_pgs)

    # --------- object (do not consider room_size_compare) ---------
    # object_ego_feats, object_phrases, object_phrase_embs, object_key_ixs, object_labels, object_masks
    object_ego_feats = np.zeros((3, self.object_seq_length, 3200), dtype=np.float32)  # (3, Lo, 3200)
    object_labels = np.zeros((3, self.object_seq_length), dtype=np.int64)  # (3, Lo)
    object_masks = np.zeros((3, self.object_seq_length), dtype=np.float32) # (3, Lo)
    object_key_ixs = np.zeros(3, dtype=np.int64)  # (3, )
    object_phrases = []
    object_phrase_embs = np.zeros((3, 300), dtype=np.float32)  # (3, 300)
    object_action_inputs = np.zeros((3, self.object_seq_length), dtype=np.int64)   # (3, Lo)
    object_action_outputs = np.zeros((3, self.object_seq_length), dtype=np.int64)  # (3, Lo)
    object_action_masks = np.zeros((3, self.object_seq_length), dtype=np.float32)  # (3, Lo)
    i = 0
    prev_end_ix = 0
    for pg, cur_key_ix in zip(nav_pgs, raw_key_ixs):
      if pg['function'] == 'nav_object':
        # next_key_ix
        next_key_ix = raw_key_ixs[raw_key_ixs.index(cur_key_ix)+1] if raw_key_ixs.index(cur_key_ix)+1 < len(raw_key_ixs) else -1
        if cur_key_ix == next_key_ix: # manually fix a bug, sometimes obj1 and obj2 are grounded in the same frame.
          next_key_ix += 1
        # prev_end_ix
        if i > 0 and nav_pgs[i-1]['function'] == 'nav_room':
          prev_end_ix = max(raw_key_ixs[i-1], prev_end_ix)
        # start_ix, end_ix for current target
        start_ix, end_ix = self.make_object_start_end(prev_end_ix, cur_key_ix, next_key_ix, self.object_seq_length, raw_path_len)
        # make object_ego_feats[i], object_labels[i], object_masks[i]
        for j, t in enumerate(range(start_ix, end_ix)):
          object_ego_feats[i][j] = raw_ego_feats[t].reshape(-1)  # (3200, )
          if t == cur_key_ix:
            object_labels[i][j] = 1  # for supervision
            object_key_ixs[i] = j    # for selection
          object_masks[i][j] = 1
          if t <= cur_key_ix:
            object_action_inputs[i][j] = raw_actions[t-1] if t > start_ix else self.act_to_ix['dummy']
            object_action_outputs[i][j] = raw_actions[t] if t < cur_key_ix else self.act_to_ix['stop']
            object_action_masks[i][j] = 1            
          if self.requires_imgs:
            object_ego_imgs[i][j] = raw_ego_imgs[t]
        # make phrases[i] and phrase_embs[i]       
        phrase = pg['value_inputs'][0]
        object_phrases.append(phrase)
        object_phrase_embs[i] = np.array([self.wtov[wd] for wd in phrase.split()]).mean(0).astype(np.float32)  # (300,)
        # next segment
        prev_end_ix = end_ix
        i += 1

    # --------- room ---------
    room_ids = []
    room_phrases = []
    room_phrase_embs = np.zeros((2, 300), dtype=np.float32)  # (2, 300)
    room_ego_feats = np.zeros((2, self.room_seq_length, 3200), dtype=np.float32)  # (2, Lr, 3200)
    room_labels = np.zeros((2, self.room_seq_length), dtype=np.int64)  # (2, Lr)
    room_masks = np.zeros((2, self.room_seq_length), dtype=np.float32) # (2, Lr)
    room_key_ixs = np.zeros(2, dtype=np.int64)  # (2, )
    room_select_ixs = np.zeros(2, dtype=np.int64)  # (2, )
    room_cube_feats = np.zeros((2, 4, 3200), dtype=np.float32)  # (2, 4, 3200)
    room_action_inputs = np.zeros((2, self.room_seq_length), dtype=np.int64)   # (2, Lr)
    room_action_outputs = np.zeros((2, self.room_seq_length), dtype=np.int64)  # (2, Lr)
    room_action_masks = np.zeros((2, self.room_seq_length), dtype=np.float32)  # (2, Lr)
    i = 0
    for pg, raw_key_ix in zip(nav_pgs, raw_key_ixs):
      if pg['function'] == 'nav_room':
        # raw_room_label
        room_id = pg['id'][0]; room_ids.append(room_id)
        raw_room_label = np.array(qn['room_labels'][room_id][pix])  # path_len, values in {-1, 0, 1}
        # start_ix, end_ix
        start_ix = random.choice(range(max(0, raw_key_ix-self.room_seq_length+1), max(1, raw_key_ix-self.pre_size))) # push a bit earlier
        end_ix = min(raw_path_len, start_ix+self.room_seq_length)
        # make room_ego_feats[i], room_labels[i], room_masks[i]
        for j, t in enumerate(range(start_ix, end_ix)):
          room_ego_feats[i][j] = raw_ego_feats[t].reshape(-1)  # (3200, )
          if t == raw_key_ix:
            room_key_ixs[i] = j
          # room_labels
          if raw_room_label[t] == 1:     # mask == label == 1 for positive 
            room_labels[i][j] = 1
            room_masks[i][j] = 1
          elif raw_room_label[t] == -1:  # mask == 1, label == 0 for negative
            room_masks[i][j] = 1
          # img
          if self.requires_imgs:
            room_ego_imgs[i][j] = raw_ego_imgs[t]
        # make room_cube_feats
        possible_room_select_ixs = np.where(raw_room_label[start_ix:end_ix] == 1)[0]
        assert possible_room_select_ixs.sum() > 0, 'there is no possitive label along this path.'
        room_select_ix = np.random.choice(possible_room_select_ixs)
        room_select_ixs[i] = room_select_ix
        room_cube_feats[i] = raw_cube_feats[start_ix:end_ix][room_select_ix].reshape(4, 3200)  # (4, 3200)
        # make room_actions based on room_select_ix
        for j, t in enumerate(range(start_ix, start_ix+room_select_ix+1)):
          room_action_inputs[i][j] = raw_actions[t-1] if t > start_ix else self.act_to_ix['dummy']
          room_action_outputs[i][j] = raw_actions[t] if t < start_ix+room_select_ix else self.act_to_ix['stop']
          room_action_masks[i][j] = 1
        if self.requires_imgs:
          room_selected_cube_imgs[i] = raw_cube_imgs[start_ix:end_ix][room_select_ix]  # (4, 224, 224, 3)
        # make phrases
        phrase = pg['value_inputs'][0]
        room_phrases.append(phrase)
        room_phrase_embs[i] = np.array([self.wtov[wd] for wd in phrase.split()]).mean(0).astype(np.float32)  # (300,)
        # next room
        i += 1

    # return
    data = {}
    data['qid'] = qid
    data['question'] = qn['question']
    data['answer'] = qn['answer']
    data['attr'] = attr
    data['room_attr'] = room_attr
    data['qe'] = qe
    data['ae'] = ae
    data['object_ego_feats'] = object_ego_feats
    data['object_phrases'] = object_phrases
    data['object_phrase_embs'] = object_phrase_embs
    data['object_key_ixs'] = object_key_ixs
    data['object_masks'] = object_masks
    data['object_labels'] = object_labels
    data['object_action_inputs'] = object_action_inputs
    data['object_action_outputs'] = object_action_outputs
    data['object_action_masks'] = object_action_masks
    data['room_ids'] = room_ids
    data['room_ego_feats'] = room_ego_feats
    data['room_phrases'] = room_phrases
    data['room_phrase_embs'] = room_phrase_embs
    data['room_cube_feats'] = room_cube_feats
    data['room_key_ixs'] = room_key_ixs
    data['room_select_ixs'] = room_select_ixs
    data['room_masks'] = room_masks
    data['room_labels'] = room_labels
    data['room_action_inputs'] = room_action_inputs
    data['room_action_outputs'] = room_action_outputs
    data['room_action_masks'] = room_action_masks
    if self.requires_imgs:
      data['object_ego_imgs'] = object_ego_imgs
      data['room_ego_imgs'] = room_ego_imgs
      data['room_selected_cube_imgs'] = room_selected_cube_imgs
    return data

  def make_object_start_end(self, prev_end_ix, cur_key_ix, next_key_ix, seq_length, path_len):
    """
    Sample start_ix and end_ix for each target
    """
    if prev_end_ix == 0:
      # first path
      assert next_key_ix != -1 and cur_key_ix < next_key_ix, 'cur_key_ix[%s], next_key_ix[%s]' % (cur_key_ix, next_key_ix)
      start_ix = self.sample(max(0, cur_key_ix-seq_length+1), max(1, cur_key_ix))
      end_ix = self.sample(cur_key_ix, min((cur_key_ix+next_key_ix) // 2 + 1, start_ix+seq_length) ) + 1  # +1 for python
      assert end_ix <= next_key_ix
    elif next_key_ix != -1:
      # second path
      start_ix = max(prev_end_ix, cur_key_ix-seq_length+1)
      end_ix = self.sample(cur_key_ix, min((cur_key_ix+next_key_ix) // 2 + 1, start_ix+seq_length) ) + 1  # +1 for python
    elif next_key_ix == -1:
      # final path
      start_ix = max(prev_end_ix, cur_key_ix-seq_length+1)
      end_ix = cur_key_ix+1
      # end_ix = self.sample(cur_key_ix+1, min(raw_path_len, cur_key_ix+seq_length))
    else:
      raise NotImplementedError    
    assert end_ix - start_ix <= seq_length, 'start_ix=%s, end_ix=%s, seq_length=%s' % (start_ix, end_ix, seq_length)
    return start_ix, end_ix

  def sample(self, ix1, ix2):
    # randomly sample from [ix1, ix2)
    assert ix1 < ix2
    return random.choice(range(ix1, ix2))

  def get_target_key_ixs(self, raw_key_ixs, nav_pgs, qn):
    assert len(nav_pgs) == len(raw_key_ixs)
    if 'object' in qn['type']:
      # targets: objects
      key_ixs = [raw_key_ixs[i] for i, nav_pg in enumerate(nav_pgs) if nav_pg['function'] == 'nav_object']
    else:
      # targets: rooms
      key_ixs = [raw_key_ixs[i] for i, nav_pg in enumerate(nav_pgs) if nav_pg['function'] == 'nav_room']
    return key_ixs

  def __len__(self):
    return len(self.questions)

