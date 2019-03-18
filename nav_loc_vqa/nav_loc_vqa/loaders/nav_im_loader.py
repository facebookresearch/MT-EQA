import os
import os.path as osp
import json
import h5py
import numpy as np
import random

import torch
from torch.utils.data import Dataset

"""
Nav Imitation Loader
We load "cache/prepro/imitation/data.json"
- questions: [{h5_id, id, question, answer, split, path_name, key_ixs_set, num_paths, bbox, program, 
               room_labels, room_inroomDists, path_actions}]
  where bbox = [{id, phrase, room_id, room_phrase, box}]
- wtoi: question vocab
- atoi: answer vocab
- ctoi: color vocab
- wtov: word2vec
We use path_name to get imgs_h5 {num_paths, orderedK, ego_rgbk, ego_semk, cube_rbbk, key_ixsk, 
                                 positionsk, actions}
and feats_h5.
"""
class NavImitationDataset(Dataset):

  def __init__(self, data_json, data_h5, 
               path_feats_dir, path_images_dir, 
               split, 
               max_seq_length=100, 
               requires_imgs=False, 
               nav_types=['all'],
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

    # actions
    self.actions = ['forward', 'left', 'right', 'dummy']
    self.act_to_ix = {'forward': 0, 'left': 1, 'right': 2, 'stop': 3, 'dummy': 4}
    self.ix_to_act = {i: a for a, i in self.act_to_ix.items()}

    # targets, whose id = path_tid
    if nav_types == ['all']:
      nav_types = ['room', 'object']
    self.targets = []
    path_tids = {}  # if path_tid has been used 
    for qn in self.questions:
      nav_pgs = [pg for pg in qn['program'] if 'nav' in pg['function']]
      assert len(nav_pgs) == len(qn['key_ixs_set'][0])
      for i, pg in enumerate(nav_pgs):
        if pg['function'][4:] in nav_types:
          tgt = {}
          tgt['qid'] = qn['id']
          tgt['house'] = qn['house']
          tgt['id'] = pg['id'][0]
          tgt['type'] = pg['function'][4:]
          tgt['phrase'] = pg['value_inputs'][0]
          tgt['key_ixs_set'] = [key_ixs[i] for key_ixs in qn['key_ixs_set']]
          if tgt['type'] == 'room':
            tgt['room_labels_set'] = qn['room_labels'][tgt['id']]
            tgt['inroomDists_set'] = qn['room_inroomDists'][tgt['id']]
          tgt['path_tid'] = '%s_%s' % (qn['path'], tgt['id'])
          if tgt['path_tid'] not in path_tids:
            self.targets.append(tgt)
            path_tids[tgt['path_tid']] = True

    print('There are in all %s %s targets.' % (len(self.targets), nav_types))
    print('Among them, %s are objects and %s are rooms.' % \
      (len([_ for _ in self.targets if _['type'] == 'object']), 
       len([_ for _ in self.targets if _['type'] == 'room'])))
    
    # more info
    self.Questions = {qn['id']: qn for qn in self.questions}
    self.split = split
    self.path_feats_dir = path_feats_dir
    self.path_images_dir = path_images_dir
    self.requires_imgs = requires_imgs
    self.pre_size = 2  # hard code this
    self.max_seq_length = max_seq_length
    self.cur_seq_length = self.max_seq_length  # used for curriculum learning

  def reset_seq_length(self, seq_length):
    self.cur_seq_length = seq_length
    assert self.cur_seq_length <= self.max_seq_length
  
  def __getitem__(self, index):
    """
    - qid
    - path_ix
    - house
    - id
    - type
    - phrase
    - phrase_emb     (300, )
    - ego_feats      (L, 3200) float32
    - next_feats     (L, 3200) float32
    - res_feats      (L, 3200) float32
    - action_inputs  (L, ) int64
    - action_outputs (L, ) int64
    - action_masks   (L, ) float32
    - ego_imgs       (L, 224, 224, 3) uint8 if necessary
    """
    tgt = self.targets[index]
    qid = tgt['qid']
    qn = self.Questions[qid]

    # choose phrase
    phrase_emb = np.array([self.wtov[wd] for wd in tgt['phrase'].split()]).mean(0).astype(np.float32)  # (300,)

    # choose path
    pix = random.choice(range(qn['num_paths'])) if self.split == 'train' else 0
    path_feats_h5 = h5py.File(osp.join(self.path_feats_dir, qn['path_name']+'.h5'), 'r')
    raw_ego_feats = path_feats_h5['ego_rgb%s' % pix]  # (raw_path_len, 32, 10, 10)
    raw_actions = qn['path_actions'][pix]  # list of path_len actions {0:'forward', 1:'left', 2:'right', 3:'stop'}
    raw_path_len = raw_ego_feats.shape[0]
    if self.requires_imgs:
      path_images_h5 = h5py.File(osp.join(self.path_images_dir, qn['path_name']+'.h5'), 'r')
      raw_ego_imgs = path_images_h5['ego_rgb%s' % pix]  # (n, 224, 224, 3)
      ego_imgs = np.zeros((self.max_seq_length, 224, 224, 3), dtype=np.uint8)  # (seq_length, 224, 224, 3)

    # ego_feats, action_inputs, action_outputs, action_masks
    ego_feats = np.zeros((self.max_seq_length, 3200), dtype=np.float32)  # (seq_length, 3200)
    next_feats = np.zeros((self.max_seq_length, 3200), dtype=np.float32) # (seq_length, 3200)
    res_feats = np.zeros((self.max_seq_length, 3200), dtype=np.float32)  # (seq_length, 3200)
    action_inputs = np.zeros(self.max_seq_length, dtype=np.int64)  # (seq_length, )
    action_outputs = np.zeros(self.max_seq_length, dtype=np.int64) # (seq_length, )
    action_masks = np.zeros(self.max_seq_length, dtype=np.float32) # (seq_length, )

    # start_ix and end_ix
    if tgt['type'] == 'object':
      key_ix = tgt['key_ixs_set'][pix]
    else:
      if self.split == 'train':
        room_labels = np.array(tgt['room_labels_set'][pix], np.int64)  # (path_len, ) {0, 1}
        possible_key_ixs = np.where(room_labels == 1)[0]
        key_ix = np.random.choice(possible_key_ixs) 
      else:
        key_ix = tgt['key_ixs_set'][pix]
    if self.split == 'train':
      start_ix = random.choice(range(max(0, key_ix-self.cur_seq_length+1), max(1, key_ix-self.pre_size)))
    else:
      start_ix = max(0, key_ix-self.max_seq_length+1)
    end_ix = key_ix + 1
    if end_ix == raw_path_len:
      end_ix -= 1
    # assert end_ix < raw_path_len, 'end_ix[%s] should be smaller than path_len[%s]' % (end_ix, raw_path_len)
    # -
    for i, t in enumerate(range(start_ix, end_ix)):
      ego_feats[i] = raw_ego_feats[t].reshape(-1)
      if t < raw_path_len-1:
        # very few case, end_ix == raw_path_len-1, we have to throw it away
        next_feats[i] = raw_ego_feats[t+1].reshape(-1)
        res_feats[i] = next_feats[i] - ego_feats[i]
      action_inputs[i] = raw_actions[t-1] if t > start_ix else self.act_to_ix['dummy']
      action_outputs[i] = raw_actions[t] if t < end_ix-1 else self.act_to_ix['stop']
      action_masks[i] = 1
      if self.requires_imgs:
        ego_imgs[i] = raw_ego_imgs[t]
    # assert action_outputs[i] == 3, action_outputs

    # return
    data = {}
    data['qid'] = qid
    data['path_ix'] = pix
    data['house'] = tgt['house']
    data['id'] = tgt['id']
    data['type'] = tgt['type']
    data['phrase'] = tgt['phrase']
    data['phrase_emb'] = phrase_emb
    data['ego_feats'] = ego_feats
    data['next_feats'] = next_feats
    data['res_feats'] = res_feats
    data['action_inputs'] = action_inputs
    data['action_outputs'] = action_outputs
    data['action_masks'] = action_masks
    if self.requires_imgs:
      data['ego_imgs'] = ego_imgs
    return data
 
  def __len__(self):
    return len(self.targets)                              


