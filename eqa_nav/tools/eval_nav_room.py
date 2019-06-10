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


def eval(rank, args, shared_nav_model, counter, split):
  # metric_names
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
    'nav_types': ['room'],
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

  while True:
    # run evaluation for this epoch
    invalids = []

    # metrics
    nav_metrics = NavMetric(
      info = {'split': split, 'thread': rank},
      metric_names = metric_names,
      log_json = args.results_json,
    )

    # sync model (fixed since now on!)
    model.load_state_dict(shared_nav_model.state_dict())
    model.eval()
    model.cuda()

    # reset envs
    eval_loader.dataset._load_envs(start_idx=0, in_order=True)

    # run
    done = False  # current epoch is not done yet
    predictions = []
    while not done:
      for batch in tqdm(eval_loader):
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
        tgt_id = batch['id'][0]
        tgt_type = batch['type'][0]
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
        for i in [5, 10, 15]:

          if action_length - i < 0:
            invalids.append((idx, i))
            continue

          h3d = eval_loader.dataset.episode_house
          episode_length = 0
          episode_done = True
          dists_to_target = []
          pos_queue = []
          actions = []

          # forward through lstm till spawn
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
            init_pos = eval_loader.dataset.episode_pos_queue[-i]
          
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
            logprobs, state = model.forward_step(ego_feat, phrase_emb, action, state)  # (1, 4), (1, rnn_size)

            # special case 1: 
            # if previous action is "forward" and collision happend and this action is still "forward", suppress it.
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
          
          # is final view looking at target object?
          if tgt_type == 'object':
            iou = h3d.compute_target_iou(tgt_id)
            R = 1 if h3d.compute_target_iou(tgt_id) >= 0.1 else 0
            pred['iou_%s' % str(i)] = iou

          # compute stats
          metrics_slug['d_0_' + str(i)] = dists_to_target[0]
          metrics_slug['ep_len_' + str(i)] = episode_length
          metrics_slug['stop_' + str(i)] = 1 if action == 3 else 0
          if tgt_type == 'object':
            metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
            metrics_slug['d_D_' + str(i)] = dists_to_target[0] - dists_to_target[-1]
            metrics_slug['d_min_' + str(i)] = float(np.array(dists_to_target).min())
            metrics_slug['h_T_' + str(i)] = R
          else:
            inside_room = []
            for p in pos_queue:
              inside_room.append(h3d.is_inside_room(p, eval_loader.dataset.target_room))
            if inside_room[-1] == True:
              metrics_slug['r_T_' + str(i)] = 1
            else:
              metrics_slug['r_T_' + str(i)] = 0
            if any([x == True for x in inside_room]) == True:
              metrics_slug['r_e_' + str(i)] = 1
            else:
              metrics_slug['r_e_' + str(i)] = 0

        # collate and update metrics
        metrics_list = []
        for i in nav_metrics.metric_names:
          if i not in metrics_slug:
            metrics_list.append(nav_metrics.metrics[nav_metrics.metric_names.index(i)][0])
          else:
            metrics_list.append(metrics_slug[i])

        # update metrics
        if len(metrics_slug) > 0:
          nav_metrics.update(metrics_list)
          predictions.append(pred)

      print(nav_metrics.get_stat_string(mode=0))
      print('invalids', len(invalids))
      logging.info("EVAL: init_steps: {} metrics: {}".format(i, nav_metrics.get_stat_string(mode=0)))
      logging.info("EVAL: init_steps: {} invalids: {}".format(i, len(invalids)))

      # next environments
      eval_loader.dataset._load_envs()
      print("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
      logging.info("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
      if len(eval_loader.dataset.pruned_env_set) == 0:
        done = True
    
    # save results
    nav_metrics.dump_log(predictions)

    # write summary
    results_str = 'id[%s] split[%s]\n' % (args.id, args.split)
    for n, r in zip(metric_names, nav_metrics.stats[-1]):
      if r:
        results_str += '%10s: %6.2f,' % (n, r[0])
        if '_15' in n: 
          results_str += '\n'
    f = open(args.report_txt, 'a')
    f.write(results_str)
    f.write('\n')
    f.close()

    # break
    break


def main(args):
  # set up model
  checkpoint_path = osp.join(args.checkpoint_dir, '%s.pth' % args.id)
  checkpoint = torch.load(checkpoint_path)
  opt = checkpoint['opt']
  model = Navigator(opt)
  model.load_state_dict(checkpoint['model_state'])
  print('model set up.')

  # evaluate
  eval(0, args, model, 0, args.split)


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  # Data input settings
  parser.add_argument('--data_json', type=str, default='cache/prepro/reinforce/data.json')
  parser.add_argument('--data_h5', type=str, default='cache/prepro/reinforce/data.h5')
  parser.add_argument('--path_feats_dir', type=str, default='cache/path_feats')
  parser.add_argument('--path_images_dir', type=str, default='cache/path_images')
  parser.add_argument('--target_obj_conn_map_dir', type=str, default='data/target-obj-conn-maps')
  parser.add_argument('--pretrained_cnn_path', type=str, default='cache/hybrid_cnn.pt')
  parser.add_argument('--house_meta_dir', type=str, default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--house_data_dir', type=str, default='data/SUNCGdata/house')
  parser.add_argument('--checkpoint_dir', type=str, default='output/nav_room')
  # Dataset settings
  parser.add_argument('--max_threads_per_gpu', type=int, default=1)
  parser.add_argument('--split', type=str, default='val')
  parser.add_argument('--max_seq_length', type=int, default=100, help='max_seq_length')
  parser.add_argument('--max_episode_length', type=int, default=50, help='maximum number of steps')
  # Output settings
  parser.add_argument('--id', type=str, default='im0')
  args = parser.parse_args()

  # make dirs
  if not osp.isdir('cache/reports'): os.makedirs('cache/reports')
  args.report_txt = 'cache/reports/nav_room.txt'
  if not osp.isdir('cache/results/nav_room'): os.makedirs('cache/results/nav_room')
  args.results_json = 'cache/results/nav_room/%s_%s.json' % (args.id, args.split)

  # run
  main(args)
