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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import _init_paths
from nav.loaders.nav_reinforce_loader import NavReinforceDataset
from nav.models.navigator import Navigator
import nav.models.utils as model_utils
from nav.models.metrics import NavMetric

import tensorboardX as tb

metric_names=[
  'd_0_5', 'd_0_10', 'd_0_15', 
  'd_T_5', 'd_T_10', 'd_T_15',
  'd_D_5', 'd_D_10', 'd_D_15', 
  'd_min_5', 'd_min_10', 'd_min_15', 
  'h_T_5', 'h_T_10', 'h_T_15',
  'r_T_5', 'r_T_10', 'r_T_15', 
  'r_e_5', 'r_e_10', 'r_e_15', 
  'stop_5', 'stop_10', 'stop_15', 
  'ep_len_5', 'ep_len_10', 'ep_len_15'
]

def eval(rank, args, shared_nav_model, counter, split='val'):
  # set up cuda device
  torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

  # tensorboard
  # writer = tb.SummaryWriter(log_dir=args.tb_dir, filename_suffix=str(rank))
  writer = tb.SummaryWriter(osp.join(args.tb_dir, str(rank)))

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
    'split': split,
    'max_seq_length': args.max_seq_length,
    'nav_types': args.nav_types,
    'gpu_id': 0,
    'cfg': cfg,
    'max_threads_per_gpu': args.max_threads_per_gpu,
    'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
    'map_resolution': 500,
    'pretrained_cnn_path': args.pretrained_cnn_path,
    'requires_imgs': False,
    'question_types': ['all'],
  }
  dataset = NavReinforceDataset(**loader_kwargs)
  eval_loader = DataLoader(dataset, batch_size=1, num_workers=0)
  print('eval_loader set up.')

  # set up model
  opt = vars(args)
  opt['act_to_ix'] = dataset.act_to_ix
  opt['num_actions'] = len(opt['act_to_ix'])
  model = Navigator(opt)
  print('eval navigator set up.')

  # evaluate
  epoch = 0
  best_eval_score = None
  while epoch < int(args.max_epochs):
    # wait till counter.value >= epoch * num_iters_per_epoch
    cur_counter_value = counter.value
    if counter.value / args.num_iters_per_epoch >= epoch:
      epoch += 1
    else:
      continue

    # run evaluation for this epoch
    invalids = []

    # metrics
    nav_metrics = NavMetric(
      info = {'split': split, 'thread': rank},
      metric_names = metric_names,
      log_json = osp.join(args.log_dir, 'nav_eval_val.json'),
    )

    # update model (fixed since now on!)
    model.load_state_dict(shared_nav_model.state_dict())
    model.eval()
    model.cuda()

    # reset envs
    eval_loader.dataset._load_envs(start_idx=0, in_order=True)
    eval_loader.dataset.visited_envs = set()

    # run
    done = False
    predictions = []
    while not done:
      for batch in tqdm(eval_loader):
        # load target_paths from available_idx (of some envs)
        # batch = {idx, qid, path_ix, house, id, type, phrase, phrase_emb, ego_feats, 
        # action_inputs, action_outputs, action_masks, action_length}
        idx = batch['idx'][0].item()
        qid = batch['qid'][0].item()
        phrase_emb = batch['phrase_emb'].cuda()  # (1, 300) cuda 
        raw_ego_feats = batch['ego_feats']       # (1, L, 3200) float
        action_inputs = batch['action_inputs']   # (1, L) int
        action_outputs = batch['action_outputs'] # (1, L) int
        action_length = batch['action_length'][0].item()
        tgt_type = batch['type'][0]
        tgt_id = batch['id'][0]
        pred = {
          'qid': qid,
          'house': batch['house'][0],
          'id': batch['id'][0],
          'type': batch['type'][0],
          'path_name': batch['path_name'][0],
          'path_ix': batch['path_ix'][0].item(),
          'start_ix': batch['start_ix'][0].item(),
          'key_ix': batch['key_ix'][0].item(),
          'action_length': action_length,
          'phrase': batch['phrase'][0],
        }
        metrics_slug = {}

        # evaluate at multiple initializations
        # for i in [5, 10, 15]:
        for i in [15]:  # for saving evaluation time

          if action_length - i < 0:
            invalids.append((idx, i))
            continue

          h3d = eval_loader.dataset.episode_house
          episode_length = 0
          episode_done = True
          dists_to_target = []
          pos_queue = []
          actions = []

          # forward through navigator till spawn
          if len(eval_loader.dataset.episode_pos_queue[:-i]):
            prev_pos_queue = eval_loader.dataset.episode_pos_queue[:-i]   # till spawned position
            # ego_imgs = eval_loader.dataset.get_frames(h3d, prev_pos_queue, preprocess=True) 
            # ego_imgs = torch.from_numpy(ego_imgs).cuda()  # (l, 3, 224, 224)
            # ego_feats = eval_loader.dataset.cnn.extract_feats(ego_imgs, conv4_only=True)  # (l, 3200)
            # ego_feats = ego_feats.view(1, len(prev_pos_queue), 3200)  # (1, l, 3200)
            ego_feats_pruned = raw_ego_feats[:, :len(prev_pos_queue), :].cuda()  # (1, l, 3200)
            action_inputs_pruned = action_inputs[:, :len(prev_pos_queue)].cuda() # (1, l)
            _, _, _, state = model(ego_feats_pruned, phrase_emb, action_inputs_pruned) # (1, l, rnn_size)
            action = action_inputs[0, len(prev_pos_queue)].view(-1).cuda() # (1, )
            init_pos = eval_loader.dataset.episode_pos_queue[-i]
          else:
            state = None            
            action = torch.LongTensor([eval_loader.dataset.act_to_ix['dummy']]).cuda() # (1, )
            init_pos = eval_loader.dataset.episode_pos_queue[0]  # use first position instead
          
          # spawn
          h3d.env.reset(x=init_pos[0], y=init_pos[2], yaw=init_pos[3])
          init_dist_to_target = h3d.get_dist_to_target(h3d.env.cam.pos)
          if init_dist_to_target < 0: # unreachable 
            invalids.append([idx, i])
            continue
          dists_to_target += [init_dist_to_target]
          pos_queue += [init_pos]

          # act
          ego_img = h3d.env.render()
          ego_img = (torch.from_numpy(ego_img.transpose(2,0,1)).float() / 255.).cuda()
          ego_feat = eval_loader.dataset.cnn.extract_feats(ego_img.unsqueeze(0), conv4_only=True)  # (1, 3200)
          collision = False
          rot_act = None
          rot_act_cnt = 0
          for step in range(args.max_episode_length):
            # forward model
            episode_length += 1
            with torch.no_grad():
              logprobs, state = model.forward_step(ego_feat, phrase_emb, action, state)  # (1, 4), (1, rnn_size)

            # special case 1: if previous action is "forward" and collision happend and this action 
            # is still "forward", suppress it.        
            if action.item() == 0 and collision and torch.exp(logprobs[0]).argmax().item() == 0:
              logprobs[0][0] = -1e5

            # special case 2:
            # if spinned around 6 times for same rotation action, we suppress it
            if torch.exp(logprobs[0]).argmax().item() == rot_act and rot_act_cnt > 5:
              logprobs[0][torch.exp(logprobs[0]).argmax().item()] = -1e5

            # sample action
            action = torch.exp(logprobs[0]).argmax().item()        
            actions += [action]

            # accumulate rot_act
            if action == 0: 
              rot_act = None
              rot_act_cnt = 0
            elif action in [1,2]:
              if rot_act == action:
                rot_act_cnt += 1
              else:
                rot_act = action
                rot_act_cnt = 1

            # interact with environment
            ego_img, _, episode_done, collision = h3d.step(action)
            episode_done = episode_done or episode_length >= args.max_episode_length  # no need actually

            # prepare state for next action
            ego_img = (torch.from_numpy(ego_img.transpose(2,0,1)).float() / 255.).cuda()
            ego_feat = eval_loader.dataset.cnn.extract_feats(ego_img.unsqueeze(0), conv4_only=True)  # (1, 3200)
            action = torch.LongTensor([action]).cuda()  # (1, )

            # add to result
            dists_to_target.append(h3d.get_dist_to_target(h3d.env.cam.pos))
            pos_queue.append([h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                              h3d.env.cam.pos.z, h3d.env.cam.yaw])

            if episode_done:
              break

          # add to predictions
          pred['d_%s' % str(i)]  = {
            'actions': actions, 
            'pos_queue': pos_queue, 
            'gd_actions': action_outputs[0, len(prev_pos_queue):len(prev_pos_queue)+i].tolist(),
            'gd_pos_queue': eval_loader.dataset.episode_pos_queue[-i:] + [eval_loader.dataset.episode_pos_queue[-1]]
          }

          # compute stats
          metrics_slug['d_0_' + str(i)] = dists_to_target[0]
          metrics_slug['ep_len_' + str(i)] = episode_length
          metrics_slug['stop_' + str(i)] = 1 if action == 3 else 0
          if tgt_type == 'object':
            metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
            metrics_slug['d_D_' + str(i)] = dists_to_target[0] - dists_to_target[-1]
            metrics_slug['d_min_' + str(i)] = float(np.array(dists_to_target).min())
            iou = h3d.compute_target_iou(tgt_id)
            metrics_slug['h_T_' + str(i)] = 1 if iou >= 0.1 else 0
          else:
            inside_room = []
            for p in pos_queue:
              inside_room.append(h3d.is_inside_room(p, eval_loader.dataset.target_room))
            metrics_slug['r_T_' + str(i)] = 1 if inside_room[-1] else 0
            metrics_slug['r_e_' + str(i)] = 1 if any([x == True for x in inside_room]) else 0
          
        # collate and update metrics
        metrics_list = []
        for name in nav_metrics.metric_names:
          if name not in metrics_slug:
            metrics_list.append(nav_metrics.metrics[nav_metrics.metric_names.index(name)][0])
          else:
            metrics_list.append(metrics_slug[name])

        # update metrics
        if len(metrics_slug) > 0:
          nav_metrics.update(metrics_list)
          predictions.append(pred)

      print(nav_metrics.get_stat_string(mode=0))
      print('invalids', len(invalids))
      logging.info("EVAL: init_steps: {} metrics: {}".format(i, nav_metrics.get_stat_string(mode=0)))
      logging.info("EVAL: init_steps: {} invalids: {}".format(i, len(invalids)))
      print('%s/%s envs visited.' % (len(eval_loader.dataset.visited_envs), len(eval_loader.dataset.env_set)))

      # next environments
      eval_loader.dataset._load_envs()
      print("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
      logging.info("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
      if len(eval_loader.dataset.pruned_env_set) == 0:
        done = True

    # write to tensorboard
    for metric_name in ['d_T_5', 'd_T_10', 'd_T_15', 'd_D_5', 'd_D_10', 'd_D_15', 
                        'h_T_5', 'h_T_10', 'h_T_15',
                        'r_T_5', 'r_T_10', 'r_T_15', 'r_e_5', 'r_e_10', 'r_e_15']:
      value = nav_metrics.metrics[nav_metrics.metric_names.index(metric_name)][0]
      if value:
        # instead of counter.value (as we started eval at cur_counter_value)
        writer.add_scalar('eval/%s' % metric_name, value, cur_counter_value)  

    # save if best
    # best_score = d_D_15 + h_T_15 + r_T_15
    cur_score = 0
    if nav_metrics.metrics[nav_metrics.metric_names.index('d_D_15')][0]:
      cur_score += nav_metrics.metrics[nav_metrics.metric_names.index('d_D_15')][0]
    if nav_metrics.metrics[nav_metrics.metric_names.index('h_T_15')][0]:
      cur_score += nav_metrics.metrics[nav_metrics.metric_names.index('h_T_15')][0]
    if nav_metrics.metrics[nav_metrics.metric_names.index('r_T_15')][0]:
      cur_score += nav_metrics.metrics[nav_metrics.metric_names.index('r_T_15')][0]
    if (best_eval_score is None or cur_score > best_eval_score) and cur_counter_value > 50000:
      best_eval_score = cur_score
      nav_metrics.dump_log(predictions)
      checkpoint_path = osp.join(args.checkpoint_dir, '%s.pth' % args.id)
      checkpoint = {}
      checkpoint['model_state'] = model.cpu().state_dict()
      checkpoint['opt'] = vars(args)
      torch.save(checkpoint, checkpoint_path)
      print('model saved to %s.' % checkpoint_path)
    
