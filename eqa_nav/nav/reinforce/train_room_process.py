# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import time
import argparse
import random
import numpy as np
import logging
import os, sys, json
import os.path as osp
from tqdm import tqdm
from runstats import Statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _init_paths
from nav.loaders.nav_reinforce_loader import NavReinforceDataset
from nav.models.navigator import Navigator
import nav.models.utils as model_utils
from nav.models.metrics import NavMetric

import tensorboardX as tb

# room-specific objects (specific version)
room_to_objects = {}
room_to_objects['bathroom'] = [ 
  'toilet', 'sink', 'shower', 'bathtub', 'towel_rack',  'mirror', 'hanger', 'partition', 'bookshelf', 
  'curtain', 'toy', 'dryer', 'heater'
  ]
room_to_objects['kitchen'] = [
  'kitchen_cabinet', 'hanging_kitchen_cabinet', 'refrigerator', 
  'microwave', 'kettle', 'coffee_machine', 'knife_rack', 'cutting_board', 'food_processor', 
  'glass', 'pan', 'plates', 'utensil_holder', 'cup', 'water_dispenser', 'range_hood', 
  'beer', 'fruit_bowl', 'dishwasher', 'range_oven'
  ]
room_to_objects['bedroom'] = [
  'double_bed', 'single_bed', 'baby_bed', 'table_lamp',  'dressing_table', 'desk', 
  'laptop', 'books', 'tv_stand', 'mirror', 'computer'
  ]
room_to_objects['living room'] = [
  'sofa', 'coffee_table', 'tv_stand', 'books', 'fireplace', 'stereo_set', 'playstation', 'clock', 
  'fish_tank', 'table_lamp', 'piano', 'xbox'
  ]
room_to_objects['dining room'] = [
  'dining_table', 'cup', 'plates', 'glass', 'candle', 'fruit_bowl', 'bottle', 'coffee_table'
  ]
room_to_objects['gym'] = [
  'gym_equipment', 'game_table', 'mirror', 'curtain', 'toy', 'outdoor_seating', 
  'vacuum_cleaner', 'basketball_hoop', 'piano', 'stereo_set', 'ironing_board', 'partition', 
  ]
room_to_objects['garage'] = [
  'car', 'garage_door', 'motorcycle', 'column', 'outdoor_lamp', 'vacuum_cleaner', 'gym_equipment', 
  'partition', 'shoes_cabinet', 'ironing_board', 
  ]
room_to_objects['balcony'] = [
  'fence', 'chair', 'outdoor_lamp', 'wall_lamp', 'grill', 'rug', 'door', 'trash_can',
  ]

def clip_model_gradient(params, grad_clip):
  for param in params:
    if hasattr(param.grad, 'data'):
      param.grad.data.clamp_(-grad_clip, grad_clip)

def ensure_shared_grads(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad

def train(rank, args, shared_nav_model, counter, lock):
  # set up tensorboard
  # writer = tb.SummaryWriter(args.tb_dir, filename_suffix=str(rank))
  writer = tb.SummaryWriter(osp.join(args.tb_dir, str(rank)))

  # set up cuda device
  gpu_id = args.gpus.index(args.gpus[rank % len(args.gpus)])
  torch.cuda.set_device(gpu_id)
  
  # set up random seeds
  random.seed(args.seed + rank)
  np.random.randn(args.seed + rank)
  torch.manual_seed(args.seed + rank)

  # set up dataset
  cfg = {
    'colorFile': osp.join(args.house_meta_dir, 'colormap_fine.csv'),
    'roomTargetFile': osp.join(args.house_meta_dir, 'room_target_object_map.csv'),
    'modelCategoryFile': osp.join(args.house_meta_dir, 'ModelCategoryMapping.csv'),
    'prefix': args.house_data_dir, 
  }
  loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'train',
    'max_seq_length': args.max_seq_length,
    'nav_types': args.nav_types,
    'gpu_id': args.gpus[rank % len(args.gpus)],
    'cfg': cfg,
    'max_threads_per_gpu': args.max_threads_per_gpu,
    'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
    'map_resolution': 500,
    'pretrained_cnn_path': args.pretrained_cnn_path,
    'requires_imgs': False,
    'question_types': ['all'],
    'ratio': [rank/args.num_processes, (rank+1)/args.num_processes]
  }
  dataset = NavReinforceDataset(**loader_kwargs)
  train_loader = DataLoader(dataset, batch_size=1, num_workers=0)
  print('train_loader set up.')

  # set up optimizer on shared_nav_model
  lr = args.learning_rate
  optimizer = torch.optim.Adam(shared_nav_model.parameters(), lr=lr, 
                               betas=(args.optim_alpha, args.optim_beta), eps=args.optim_epsilon,
                               weight_decay=args.weight_decay)

  # set up model
  opt = vars(args)
  opt['act_to_ix'] = dataset.act_to_ix
  opt['num_actions'] = len(opt['act_to_ix'])
  model = Navigator(opt)
  print('navigator[%s] set up.' % rank)

  # set up metrics outside epoch, as we use running rewards through whole training
  nav_metrics = NavMetric(
    info={'split': 'train', 'thread': rank},
    metric_names=['reward', 'episode_length'],
    log_json=osp.join(args.log_dir, 'nav_train_'+str(rank) + '.json'),
  )
  nav_metrics.update([0, 100])
  reward_list, episode_length_list = [], []
  rwd_stats = Statistics()  # computing running mean and std of rewards
  
  # path length multiplier
  min_dist = 5 if args.nav_types == ['object'] else 15  # 15 for rl3 and rl5, 10 for rl4
  # max_dist = 25 if args.nav_types == ['object'] else 40
  max_dist = 35 if args.nav_types == ['object'] else 50
  mult = 0.1 if args.nav_types == ['object'] else 0.15
  rwd_thresh = 0.1 if args.nav_types == ['object'] else 0.00
  epoch = 0
  iters = 0

  # train
  while True:
    
    # reset envs
    train_loader.dataset._load_envs(start_idx=0, in_order=True)
    done = False  # current epoch is not done yet
    while not done:

      for batch in train_loader:
        # sync model
        model.load_state_dict(shared_nav_model.state_dict())
        model.train()
        model.cuda()

        # load target_paths from available_idx (of some envs)
        # batch = {idx, qid, path_ix, house, id, type, phrase, phrase_emb, ego_feats, 
        # action_inputs, action_outputs, action_masks, action_length}
        idx = batch['idx'][0].item()
        qid = batch['qid'][0].item()
        phrase_emb = batch['phrase_emb'].cuda()  # (1, 300) float
        raw_ego_feats = batch['ego_feats']       # (1, L, 3200) float
        action_inputs = batch['action_inputs']   # (1, L) int
        action_outputs = batch['action_outputs'] # (1, L) int
        action_length = batch['action_length'][0].item()
        tgt_type = batch['type'][0]
        tgt_id = batch['id'][0]
        tgt_phrase = batch['phrase'][0]

        # to be recorded
        episode_length = 0
        episode_done = True
        dists_to_target = []
        pos_queue = []
        actions = []
        rewards = []
        nav_log_probs = []

        # spawn agent 
        h3d = train_loader.dataset.episode_house
        if np.random.uniform(0, 1, 1)[0] <= args.shortest_path_ratio:
          # half chance we use shortest path to spawn the agent (if vlen > 0, i.e., shortest path long enough)
          use_shortest_path = True
          vlen = min(max(min_dist, int(mult * action_length)), action_length)          
          # forward throught navigator till spawn
          if len(train_loader.dataset.episode_pos_queue) > vlen:
            prev_pos_queue = train_loader.dataset.episode_pos_queue[:-vlen]      # till spawned position
            ego_feats_pruned = raw_ego_feats[:, :len(prev_pos_queue), :].cuda()  # (1, l, 3200)
            action_inputs_pruned = action_inputs[:, :len(prev_pos_queue)].cuda() # (1, l)
            _, _, _, state = model(ego_feats_pruned, phrase_emb, action_inputs_pruned) # (1, l, rnn_size)
            action = action_inputs[0, len(prev_pos_queue)].view(-1).cuda() # (1, )
            init_pos = train_loader.dataset.episode_pos_queue[-vlen]
          else:
            state = None            
            action = torch.LongTensor([train_loader.dataset.act_to_ix['dummy']]).cuda() # (1, )
            init_pos = train_loader.dataset.episode_pos_queue[0]  # use first position of the path
        else:
          # half chance we randomly spawn agent 
          use_shortest_path = False
          state = None
          action = torch.LongTensor([train_loader.dataset.act_to_ix['dummy']]).cuda() # (1, )
          init_pos, vlen = train_loader.dataset.spawn_agent(min_dist, max_dist)
          if init_pos is None:  # init_pos not found
            continue

        # initiate
        h3d.env.reset(x=init_pos[0], y=init_pos[2], yaw=init_pos[3])
        init_dist_to_target = h3d.get_dist_to_target(h3d.env.cam.pos)
        if init_dist_to_target < 0: # unreachable 
          continue
        dists_to_target += [init_dist_to_target]
        pos_queue += [init_pos]

        # act
        ego_img = h3d.env.render()
        ego_img = (torch.from_numpy(ego_img.transpose(2,0,1)).float() / 255.).cuda()
        ego_feat = train_loader.dataset.cnn.extract_feats(ego_img.unsqueeze(0), conv4_only=True)  # (1, 3200)
        prev_action, collision = None, False
        for step in range(args.max_episode_length):
          # forward model one step
          episode_length += 1
          logprobs, state = model.forward_step(ego_feat, phrase_emb, action, state)  # (1, 4), (1, rnn_size)

          # sample action
          probs = torch.exp(logprobs)  # (1, 4)
          action = probs.multinomial(num_samples=1).detach() # (1, 1)
          if prev_action == 0 and collision and action[0][0].item() == 0:
            # special case: prev_action == "forward" && collision && cur_action == "forward"
            # we sample from {'left', 'right', 'stop'} only
            action = probs[0:1, 1:].multinomial(num_samples=1).detach() + 1  # (1, 1)

          if len(pos_queue) < min_dist:
            # special case: our room navigator tends to stop early, let's push it to explore longer
            action = probs[0:1, :3].multinomial(num_samples=1).detach()  # (1, 1)

          nav_log_probs.append(logprobs.gather(1, action))   # (1, 1)
          action = action.view(-1)     # (1, )
          actions.append(action)

          # interact with environment
          ego_img, reward, episode_done, collision = h3d.step(action[0].item(), step_reward=True)
          if not episode_done:
            reward -= 0.01  # we don't wanna too long trajectory
          episode_done = episode_done or episode_length >= args.max_episode_length  # no need actually
          reward = max(min(reward, 1), -1)
          rewards.append(reward)
          prev_action = action[0].item()

          # prepare state for next action
          ego_img = (torch.from_numpy(ego_img.transpose(2,0,1)).float() / 255.).cuda()
          ego_feat = train_loader.dataset.cnn.extract_feats(ego_img.unsqueeze(0), conv4_only=True)  # (1, 3200)

          # add to result
          dists_to_target.append(h3d.get_dist_to_target(h3d.env.cam.pos))
          pos_queue.append([h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                            h3d.env.cam.pos.z, h3d.env.cam.yaw])

          if episode_done:
            break

        # final reward
        R = 0
        if tgt_type == 'object':
          R = 1.0 if h3d.compute_target_iou(tgt_id) >= 0.1 else -1.0
        else: 
          R = 0.2 if h3d.is_inside_room(pos_queue[-1], train_loader.dataset.target_room) else -0.2
          if R > 0 and h3d.compute_room_targets_iou(room_to_objects[tgt_phrase]) > 0.1:
            R += 1.0  # encourage agent to move to room-specific targets

        # backward 
        nav_loss = 0
        new_rewards = []  # recording reshaped rewards, to be computed as moving average
        advantages = []
        for i in reversed(range(len(rewards))):
          R = 0.99 * R + rewards[i]
          new_rewards.insert(0, R)
          rwd_stats.push(R)
        for nav_log_prob, R in zip(nav_log_probs, new_rewards):
          advantage = (R - rwd_stats.mean()) / (rwd_stats.stddev() + 1e-5)  # rl2, rl3
          # advantage = R - rwd_stats.mean()  # rl1
          nav_loss = nav_loss - nav_log_prob * advantage
          advantages.insert(0, advantage)
        nav_loss /= max(1, len(nav_log_probs))

        optimizer.zero_grad()
        nav_loss.backward()
        clip_model_gradient(model.parameters(), args.grad_clip)
        # Till this point, we have grads on model's parameters!
        # but shared_nav_model's grads is still not updated 
        ensure_shared_grads(model.cpu(), shared_nav_model)
        optimizer.step()

        if iters % 5 == 0:
          log_info = 'train-r%2s(ep%2s it%5s lr%.2E mult%.2f, run_rwd%6.3f, bs_rwd%6.3f, bs_std%2.2f), vlen(%s)=%2s, rwd=%6.3f, ep_len=%3s, tgt:%s' % \
            (rank, epoch, iters, lr, mult, nav_metrics.metrics[0][1], rwd_stats.mean(), rwd_stats.stddev(), 'short' if use_shortest_path else 'spawn', vlen, np.mean(new_rewards), len(new_rewards), tgt_phrase)
          print(log_info)

        # update metrics
        reward_list += new_rewards
        episode_length_list.append(episode_length)
        if len(episode_length_list) > 50:
          nav_metrics.update([reward_list, episode_length_list])
          nav_metrics.dump_log()
          reward_list, episode_length_list = [], []

        # write to tensorboard
        writer.add_scalar('train_rank%s/nav_loss'%rank, nav_loss.item(), counter.value)
        writer.add_scalar('train_rank%s/steps_avg_rwd'%rank, float(np.mean(new_rewards)), counter.value)
        writer.add_scalar('train_rank%s/steps_sum_rwd'%rank, float(np.sum(new_rewards)), counter.value)
        writer.add_scalar('train_rank%s/advantage'%rank, float(np.mean(advantages)), counter.value)

        # increase counter as this episode ends
        with lock:
          counter.value += 1

        # increase mult
        iters += 1
        if nav_metrics.metrics[0][1] > rwd_thresh:  # baseline = nav_metrics.metrics[0][1]
          mult = min(mult + 0.1, 1.0)
          rwd_thresh += 0.01
        else:
          mult = max(mult - 0.1, 0.1)
          rwd_thresh -= 0.01
        rwd_thresh = max(0.1, min(rwd_thresh, 0.2))

        # decay learning rate
        if args.learning_rate_decay_start > 0 and iters > args.learning_rate_decay_start:
          frac = (iters - args.learning_rate_decay_start) / args.learning_rate_decay_every
          decay_factor = 0.1 ** frac
          lr = args.learning_rate * decay_factor
          model_utils.set_lr(optimizer, lr)

      # next environments
      train_loader.dataset._load_envs(in_order=True)
      print("train_loader pruned_env_set len: {}".format(len(train_loader.dataset.pruned_env_set)))
      logging.info("train_loader pruned_env_set len: {}".format(len(train_loader.dataset.pruned_env_set)))
      if len(train_loader.dataset.pruned_env_set) == 0:
        done = True

    epoch += 1
