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

from nav_loc_vqa.models.cnn import MultitaskCNN

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
class NavLocVqaRlDataset(Dataset):
  def __init__(self, data_json, data_h5,
               path_feats_dir,
               path_images_dir,
               split,
               gpu_id,
               max_threads_per_gpu,
               cfg,
               to_cache,
               target_obj_conn_map_dir,
               map_resolution,
               pretrained_cnn_path,
               requires_imgs=False,
               question_types=['all'],
               ratio=None, 
               height=224,
               width=224,
               num_questions=-1):

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

    # questions
    if question_types == ['all']:
      self.questions = [qn for qn in self.infos['questions'] if qn['split'] == split]
    else:
      self.questions = [qn for qn in self.infos['questions'] if qn['split'] == split and qn['type'] in question_types]
    if num_questions != -1:
      self.questions = self.questions[:num_questions]
    self.Questions = {qn['id']: qn for qn in self.questions}
    self.ids = [qn['id'] for qn in self.questions]
    print('%s questions loaded for type%s under split[%s].' % (len(self.questions), question_types, split))

    # hid_tid_to_best_iou
    self.hid_tid_to_best_iou = self.infos['hid_tid_to_best_iou']  # hid_tid --> best_iou

    # load data.h5
    encoded = h5py.File(data_h5, 'r')
    self.encoded_questions = encoded['encoded_questions']
    self.encoded_answers = encoded['encoded_answers']
    assert self.encoded_questions.shape[0] == self.encoded_answers.shape[0]
    print('max_length of encoded_questions is', self.encoded_questions.shape[1])
    print('[%s] data prepared, where there are %s questions.' % (split, len(self.questions)))

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

    self.gpu_id = gpu_id
    self.cfg = cfg
    self.max_threads_per_gpu = max_threads_per_gpu
    self.target_obj_conn_map_dir = target_obj_conn_map_dir
    self.map_resolution = map_resolution
    self.to_cache = to_cache
    self.height = height or 224
    self.width = width or 224

    self.episode_pos_queue = None
    self.episode_house = None
    self.target_room = None
    self.target_obj = None
    self.img_data_cache = {}   # qid --> feats
    self.available_idx = []
    self.visited_envs = set()
    self.api_threads = []

    # set up cnn
    cnn_kwargs = {'num_classes': 191, 'pretrained': True, 'checkpoint_path': pretrained_cnn_path}
    self.cnn = MultitaskCNN(**cnn_kwargs).cuda()
    self.cnn.eval()
    print('cnn set up.')

    # construct mapping
    self.envs = list(set([qn['house'] for qn in self.questions]))  # all house_ids
    self.env_idx = [self.envs.index(qn['house']) for qn in self.questions]  # house index for each question
    self.env_list = [self.envs[x] for x in self.env_idx]  # list of house_ids for each question
    self.env_set = list(set(self.env_list))  
    self.env_set.sort()  # ordered house_ids
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
        self.api_threads.append(objrender.RenderAPIThread(w=self.width, h=self.height, device=self.gpu_id))
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
    - qid, house
    - question, answer
    - qe, ae
    - type
    - attr
    - path_ix
    - nav_ids
    - nav_types
    - nav_ego_feats        #navs of (l, 3200) float32
    - nav_action_inputs    #navs of (l, ) int64
    - nav_action_outputs   #navs of (l, ) int64
    - nav_ego_imgs         #navs of (l, 224, 224, 3) uint8 if necessary
    private variables:
    - episode_house  
    - nav_pos_queues       #navs of l [x, y, z, yaw]
    - path_len
    """
    idx = self.available_idx[index]
    qn = self.questions[idx]
    qid = qn['id']
    house = qn['house']
    attr, room_attr = self.question_to_attribute(qn)

    # encode question and answer
    qe = self.encoded_questions[qn['h5_id']]
    ae = self.encoded_questions[qn['h5_id']]

    # choose path_ix
    path_ix = random.choice(range(qn['num_paths'])) if self.split == 'train' else 0 
    path_feats_h5 = h5py.File(osp.join(self.path_feats_dir, qn['path_name']+'.h5'), 'r')
    raw_ego_feats = path_feats_h5['ego_rgb%s' % path_ix][...].reshape(-1, 3200)  # (L, 32, 10, 10)
    raw_path_len = raw_ego_feats.shape[0]
    raw_actions = qn['path_actions'][path_ix]  # (L, )
    raw_pos_queue = qn['path_positions'][path_ix]  # list of L positions
    if self.requires_imgs:
      path_images_h5 = h5py.File(osp.join(self.path_images_dir, qn['path_name']+'.h5'), 'r')
      raw_ego_imgs = path_images_h5['ego_rgb%s' % path_ix]  # (L, 224, 224, 3)
      nav_ego_imgs = []

    # nav_phrases, nav_phrase_embs
    nav_pgs = [pg for pg in qn['program'] if 'nav' in pg['function']]
    nav_ids = [pg['id'][0] for pg in qn['program'] if 'nav' in pg['function']]
    nav_types = [pg['function'][4:] for pg in qn['program'] if 'nav' in pg['function']]
    nav_phrases = [pg['value_inputs'][0] for pg in qn['program'] if 'nav' in pg['function']]
    nav_phrase_embs = []
    for phrase in nav_phrases:
      nav_phrase_embs.append(np.array([self.wtov[wd] for wd in phrase.split()]).mean(0).astype(np.float32))  # (300, )
    nav_phrase_embs = np.array(nav_phrase_embs)  # (#targets, 300)

    # For each segment path: feats + actions + pos_queue
    raw_key_ixs = qn['key_ixs_set'][path_ix]
    nav_ego_feats = []
    nav_action_inputs = []
    nav_action_outputs = []
    nav_pos_queues = []
    for i, key_ix in enumerate(raw_key_ixs):
      start_ix = 0 if i == 0 else raw_key_ixs[i-1]  # we use last key_ix moment as start (spawn location)
      end_ix = raw_key_ixs[i]+1
      ego_feats = raw_ego_feats[start_ix:end_ix]
      action_inputs = np.array([4] + raw_actions[start_ix:end_ix][:-1], dtype=np.int64)
      action_outputs = np.array(raw_actions[start_ix:end_ix-1] + [3], dtype=np.int64)
      pos_queue = raw_pos_queue[start_ix:end_ix]
      assert ego_feats.shape[0] == len(pos_queue) == action_inputs.shape[0]
      # add to list
      nav_ego_feats.append(ego_feats)
      nav_action_inputs.append(action_inputs)
      nav_action_outputs.append(action_outputs)
      nav_pos_queues.append(pos_queue)
      if self.requires_imgs:
        nav_ego_imgs.append(raw_ego_imgs[start_ix:end_ix])

    # cache
    if self.to_cache and index not in self.img_data_cache:
      self.img_data_cache[index] = True  # TODO: replace with ego_feats 

    # private variable
    self.episode_house = self.env_loaded[house]
    self.nav_pos_queues = nav_pos_queues
    self.path_len = raw_path_len

    # return
    data = {}
    data['idx'] = idx
    data['qid'] = qid
    data['house'] = qn['house']
    data['question'] = qn['question']
    data['answer'] = qn['answer']
    data['type'] = qn['type']
    data['attr'] = attr
    data['qe'] = qe
    data['ae'] = ae
    data['path_name'] = qn['path_name']
    data['path_ix'] = path_ix
    data['nav_ids'] = nav_ids
    data['nav_types'] = nav_types
    data['nav_phrases'] = nav_phrases
    data['nav_phrase_embs'] = nav_phrase_embs
    data['nav_ego_feats'] = nav_ego_feats
    data['nav_action_inputs'] = nav_action_inputs
    data['nav_action_outputs'] = nav_action_outputs
    if self.requires_imgs:
      data['nav_ego_imgs'] = nav_ego_imgs
    return data

  def spawn_agent(self, min_dist, max_dist, split):
    """
    Run set_target_object/room before calling this function!
    Inputs: 
    - min_dist/max_dist: distance between target and spawned location (in connMap)
    Return:
    - position
    - distance
    """
    conn_map = self.episode_house.env.house.connMap
    point_cands = np.argwhere((conn_map > min_dist) & (conn_map <= max_dist) )
    if point_cands.shape[0] == 0:
      return None, None
    point_idx = np.random.choice(point_cands.shape[0]) if split == 'train' else 0 # 0 for inference
    point = (point_cands[point_idx][0], point_cands[point_idx][1])
    gx, gy = self.episode_house.env.house.to_coor(point[0], point[1])
    yaw = np.random.choice(self.episode_house.angles) if split == 'train' else 0  # 0 for inference
    return [float(gx), 1.0, float(gy), float(yaw)], conn_map[point]

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

  def __len__(self):
    return len(self.available_idx)
