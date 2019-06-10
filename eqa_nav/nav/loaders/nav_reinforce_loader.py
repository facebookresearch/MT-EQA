# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import h5py
import argparse
import numpy as np
import os, sys, json
import os.path as osp
import random

import torch
from torch.utils.data import Dataset

from House3D import objrender, Environment, load_config
from House3D.core import local_create_house

from house3d import House3DUtils

from nav.models.cnn import MultitaskCNN

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
class NavReinforceDataset(Dataset):
  def __init__(self, data_json, data_h5,
               path_feats_dir,
               path_images_dir,
               split,
               max_seq_length,
               nav_types,
               gpu_id,
               max_threads_per_gpu,
               cfg,
               to_cache=False,
               target_obj_conn_map_dir=False,
               map_resolution=500,
               pretrained_cnn_path='cache/hybrid_cnn.pt',
               requires_imgs=False,
               question_types=['all'],
               ratio=None):
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

    # actions
    self.act_to_ix = {'forward': 0, 'left': 1, 'right': 2, 'stop': 3, 'dummy': 4}
    self.ix_to_act = {i: a for a, i in self.act_to_ix.items()}

    # questions
    if question_types == ['all']:
      self.questions = [qn for qn in self.infos['questions'] if qn['split'] == split]
    else:
      self.questions = [qn for qn in self.infos['questions'] if qn['split'] == split and qn['type'] in question_types]
    self.Questions = {qn['id']: qn for qn in self.questions}
    print('%s questions loaded for type%s under split[%s].' % (len(self.questions), question_types, split))

    # hid_tid_to_best_iou
    self.hid_tid_to_best_iou = self.infos['hid_tid_to_best_iou']  # hid_tid --> best_iou

    # more info
    self.split = split
    self.gpu_id = gpu_id
    self.cfg = cfg
    self.max_threads_per_gpu = max_threads_per_gpu
    self.target_obj_conn_map_dir = target_obj_conn_map_dir
    self.map_resolution = map_resolution
    self.to_cache = to_cache
    self.path_feats_dir = path_feats_dir
    self.path_images_dir = path_images_dir
    self.requires_imgs = requires_imgs
    self.pre_size = 2  # hard code this
    self.max_seq_length = max_seq_length
    
    self.episode_pos_queue = None
    self.episode_house = None
    self.target_room = None
    self.target_obj = None
    self.img_data_cache = {}   # hid.tid --> feats
    self.available_idx = []
    self.visited_envs = set()
    self.api_threads = []

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
          if tgt['type'] == 'object':
            tgt['fine_class'] = pg['fine_class'][0]
          if tgt['type'] == 'room':
            tgt['room_labels_set'] = qn['room_labels'][tgt['id']]
            tgt['inroomDists_set'] = qn['room_inroomDists'][tgt['id']]
          tgt['path_tid'] = '%s_%s' % (qn['path'], tgt['id'])
          if tgt['path_tid'] not in path_tids:
            self.targets.append(tgt)
            path_tids[tgt['path_tid']] = True

    if split in ['val', 'test']:
      # otherwise too many evaluations...cut to half! TODO: change back for whole evaluation
      self.targets = self.targets[:len(self.targets) // 3]  
    print('There are in all %s %s targets.' % (len(self.targets), nav_types))
    print('Among them, %s are objects and %s are rooms.' % \
      (len([_ for _ in self.targets if _['type'] == 'object']), 
       len([_ for _ in self.targets if _['type'] == 'room'])))

    # set up cnn
    cnn_kwargs = {'num_classes': 191, 'pretrained': True, 'checkpoint_path': pretrained_cnn_path}
    self.cnn = MultitaskCNN(**cnn_kwargs).cuda()
    self.cnn.eval()
    print('cnn set up.')

    # construct mapping
    self.envs = list(set([tgt['house']for tgt in self.targets]))  # all house_ids 
    self.env_idx = [self.envs.index(tgt['house']) for tgt in self.targets]  # house index for each target
    self.env_list = [self.envs[x] for x in self.env_idx]  # list of house_ids for all targets
    self.env_set = list(set(self.env_list))  # all house_ids, should be same as envs
    self.env_set.sort()
    if ratio:
      assert isinstance(ratio, list), ratio
      self.env_set = self.env_set[int(ratio[0]*len(self.env_set)):int(ratio[1]*len(self.env_set))]
    print('Total envs: %d' % len(self.envs))
    print('Envs in [%s]: %d, we use %d.' % (self.split, len(list(set(self.env_idx))), len(self.env_set)))

    # load environments
    self._load_envs(start_idx=0, in_order=True)

  def _pick_envs_to_load(self, split, max_envs, start_idx, in_order):
    """
    pick houses from self.env_set
    """
    if split in ['val', 'test'] or in_order:
      pruned_env_set = self.env_set[start_idx:start_idx+max_envs]  # could be void if start_idx arrives end
    else:
      if max_envs < len(self.env_set):
        env_inds = np.random.choice(len(self.env_set), max_envs, replace=False) 
      else:
        env_inds = np.random.choice(len(self.env_set), max_envs, replace=True)
      pruned_env_set = [self.env_set[x] for x in env_inds]
    return pruned_env_set 
  
  def _load_envs(self, start_idx=-1, in_order=False):
    if start_idx == -1:  # next env
      start_idx = self.env_set.index(self.pruned_env_set[-1]) + 1

    # pick envs
    self.pruned_env_set = self._pick_envs_to_load(self.split, self.max_threads_per_gpu, 
                                                  start_idx, in_order)
    if len(self.pruned_env_set) == 0:
      return

    # Load api threads
    start = time.time()
    if len(self.api_threads) == 0:
      for i in range(self.max_threads_per_gpu):
        self.api_threads.append(objrender.RenderAPIThread(w=224, h=224, device=self.gpu_id))
    print('[%.2f] Loaded %d api threads' % (time.time()-start, len(self.api_threads)))

    # Load houses
    start = time.time()
    from multiprocessing import Pool
    _args = ([h, self.cfg, self.map_resolution] for h in self.pruned_env_set)
    with Pool(len(self.pruned_env_set)) as pool:
      self.all_houses = pool.starmap(local_create_house, _args)
    print('[%.02f] Loaded %d houses' % (time.time() - start, len(self.all_houses)))

    # Load envs 
    start = time.time()
    self.env_loaded = {}
    for i in range(len(self.all_houses)):
      print('[%02d/%d][split:%s][gpu:%d][house:%s]' %
        (i + 1, len(self.all_houses), self.split, self.gpu_id, self.all_houses[i].house['id']))
      env = Environment(self.api_threads[i], self.all_houses[i], self.cfg)
      self.env_loaded[self.all_houses[i].house['id']] = \
        House3DUtils(env, target_obj_conn_map_dir=self.target_obj_conn_map_dir)
    print('[%.02f] Loaded %d house3d envs' % (time.time() - start, len(self.env_loaded)))

    for i in range(len(self.all_houses)):
      self.visited_envs.add(self.all_houses[i].house['id'])

    # Mark available data indices
    self.available_idx = [i for i, v in enumerate(self.env_list) if v in self.env_loaded]
    print('Available inds: %d' % len(self.available_idx))

  def _load_env(self, house):
    # For testing (ipynb) only, we wanna load just one house.
    start = time.time()
    self.all_houses = [local_create_house(house, self.cfg, self.map_resolution)]
    env = Environment(self.api_threads[0], self.all_houses[0], self.cfg)
    self.env_loaded[house] = House3DUtils(env, target_obj_conn_map_dir=self.target_obj_conn_map_dir)
    print('[%.02f] Loaded 1 house3d envs' % (time.time() - start))

  def _check_if_all_envs_loaded(self):
    print('[CHECK][Visited:%d envs][Total:%d envs]' % (len(self.visited_envs), len(self.env_set)))
    return True if len(self.visited_envs) == len(self.env_set) else False

  def _check_if_all_targets_loaded(self):
    print('[CHECK][Visited:%d targets][Total:%d targets]' % (len(self.img_data_cache), len(self.env_list)))
    if len(self.img_data_cache) == len(self.env_list):
      self.available_idx = [i for i, v in enumerate(self.env_list)]
      return True
    else:
      return False

  def set_camera(self, e, pos, robot_height=1.0):
    assert len(pos) == 4
    e.env.cam.pos.x = pos[0]
    e.env.cam.pos.y = robot_height
    e.env.cam.pos.z = pos[2]
    e.env.cam.yaw = pos[3]
    e.env.cam.updateDirection()
  
  def render(self, e):
    return e.env.render()
  
  def get_frames(self, e, pos_queue, preprocess=True):
    # return imgs (n, 3, 224, 224) along pos_queue
    if not isinstance(pos_queue, list):
      pos_queue = [pos_queue]
    
    res = []
    for i in range(len(pos_queue)):
      self.set_camera(e, pos_queue[i])
      img = np.array(self.render(e), copy=False, dtype=np.uint8)
      if preprocess:
        img = img.astype(np.float32) / 255.
        img = img.transpose(2, 0, 1)  # (3, 224, 224)
      res.append(img) 
    return np.array(res)

  def __getitem__(self, index):
    """
    - idx
    - qid
    - path_ix
    - house
    - id
    - type
    - phrase
    - phrase_emb
    - ego_feats       (L, 3200) float32
    - action_inputs   (L, ) int64
    - action_outputs  (L, ) int64
    - action_masks    (L, ) float32
    - ego_imgs        (L, 224, 224, 3) uint8 if necessary
    """
    idx = self.available_idx[index]
    tgt = self.targets[idx]
    qid = tgt['qid']
    hid = tgt['house']
    qn = self.Questions[qid]
    assert tgt['house'] == self.env_list[idx]

    # choose path_ix
    pix = random.choice(range(qn['num_paths'])) if self.split == 'train' else 0
    path_feats_h5 = h5py.File(osp.join(self.path_feats_dir, qn['path_name']+'.h5'), 'r')
    raw_ego_feats = path_feats_h5['ego_rgb%s' % pix]  # (raw_path_len, 32, 10, 10)
    raw_actions = qn['path_actions'][pix]  # list of path_len actions {0:'forward', 1:'left', 2:'right', 3:'stop'}
    raw_path_len = raw_ego_feats.shape[0]
    if self.requires_imgs:
      path_images_h5 = h5py.File(osp.join(self.path_images_dir, qn['path_name']+'.h5'), 'r')
      raw_ego_imgs = path_images_h5['ego_rgb%s' % pix]  # (n, 224, 224, 3)
      ego_imgs = np.zeros((self.max_seq_length, 224, 224, 3), dtype=np.uint8)  # (seq_length, 224, 224, 3)

    # phrase
    phrase_emb = np.array([self.wtov[wd] for wd in tgt['phrase'].split()]).mean(0).astype(np.float32)  # (300,)

    # ego_feats, action_inputs, action_outputs, action_masks
    ego_feats = np.zeros((self.max_seq_length, 3200), dtype=np.float32)  # (seq_length, 3200)
    action_inputs = np.zeros(self.max_seq_length, dtype=np.int64)  # (seq_length, )
    action_outputs = np.zeros(self.max_seq_length, dtype=np.int64) # (seq_length, )
    action_masks = np.zeros(self.max_seq_length, dtype=np.float32) # (seq_length, )

    # start and end along shortest path
    self.episode_house = self.env_loaded[hid]
    self.target_obj = None
    self.target_room = None
    if tgt['type'] == 'object':
      # target object
      self.target_obj = self.env_loaded[hid].objects[tgt['id']]
      self.episode_house.set_target_object(self.target_obj)
      # shortest path 
      key_ix = tgt['key_ixs_set'][pix]
      start_ix = max(0, key_ix-self.max_seq_length+1)
      end_ix = key_ix + 1
      if end_ix == raw_path_len:
        end_ix -= 1
    else:
      # target room
      self.target_room = self.episode_house.rooms[tgt['id']]
      self.episode_house.set_target_room(self.target_room)
      # we will check since what position agent starts lying inside room, chunk its follow-up.
      inside_cnt = 0
      for t, pos in enumerate(qn['path_positions'][pix]):
        if not self.episode_house.is_inside_room(pos, self.target_room):
          inside_cnt = 0
        else:
          inside_cnt += 1
          if inside_cnt > 2:  # allow 3 positions to be inside room to ease "door"/"wall" obstacle issue
            break
      end_ix = min(max(5, t), len(qn['path_positions'][pix]))  # we wanna the path to be at least 5 actions
      start_ix = max(0, end_ix-self.max_seq_length) 

    # feats, actions, positions
    for i, t in enumerate(range(start_ix, end_ix)):
      ego_feats[i] = raw_ego_feats[t].reshape(-1)
      action_inputs[i] = raw_actions[t-1] if t > start_ix else self.act_to_ix['dummy']
      action_outputs[i] = raw_actions[t] if t < end_ix-1 else self.act_to_ix['stop']
      action_masks[i] = 1
      if self.requires_imgs:
        ego_imgs[i] = raw_ego_imgs[t]
    assert action_outputs[i] == 3, action_outputs
    # positions
    self.episode_pos_queue = qn['path_positions'][pix][start_ix:end_ix]  # list of action_length [pos (x, y, z, yaw)]
    assert len(self.episode_pos_queue) == int(np.sum(action_masks)), 'pos_queue%s, action_length%s' % (len(self.episode_pos_queue), int(np.sum(action_masks)))

    # cache
    if self.to_cache and index not in self.img_data_cache:
      self.img_data_cache[index] = True  # TODO: replace with ego_feats

    # return  
    data = {}
    data['idx'] = idx
    data['qid'] = qid
    data['path_name'] = qn['path_name']
    data['path_ix'] = pix
    data['start_ix'] = start_ix
    data['key_ix'] = end_ix-1
    data['house'] = tgt['house']
    data['id'] = tgt['id']
    data['type'] = tgt['type']
    data['phrase'] = tgt['phrase']
    data['phrase_emb'] = phrase_emb
    data['ego_feats'] = ego_feats
    data['action_inputs'] = action_inputs
    data['action_outputs'] = action_outputs
    data['action_masks'] = action_masks
    data['action_length'] = int(np.sum(action_masks))
    if self.requires_imgs:
      data['ego_imgs'] = ego_imgs
    return data
  
  def spawn_agent(self, min_dist, max_dist):
    """
    dist: distance between target and spawned location (in connMap)
    """
    conn_map = self.episode_house.env.house.connMap
    point_cands = np.argwhere((conn_map > min_dist) & (conn_map <= max_dist) )
    if point_cands.shape[0] == 0:
      return None, None
    point_idx = np.random.choice(point_cands.shape[0])
    point = (point_cands[point_idx][0], point_cands[point_idx][1])
    gx, gy = self.episode_house.env.house.to_coor(point[0], point[1])
    yaw = np.random.choice(self.episode_house.angles)
    return [float(gx), 1.0, float(gy), float(yaw)], conn_map[point]
 
  def __len__(self):
    return len(self.available_idx)


