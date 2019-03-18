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
from nav_loc_vqa.models.crits import SeqModelCriterion
from nav_loc_vqa.loaders.nav_im_loader import NavImitationDataset
from nav_loc_vqa.loaders.nav_loc_vqa_rl_loader import NavLocVqaRlDataset 
from nav_loc_vqa.models.navigator import Navigator
from nav_loc_vqa.models.factory import ModelFactory 
import nav_loc_vqa.models.utils as model_utils
from nav_loc_vqa.models.metrics import Metric

# metric_names
nav_object_metric_names=[
  'd_0_5', 'd_0_10', 'd_0_15', 
  'd_T_5', 'd_T_10', 'd_T_15',
  'd_D_5', 'd_D_10', 'd_D_15', 
  'd_min_5', 'd_min_10', 'd_min_15', 
  'h_T_5a', 'h_T_10a', 'h_T_15a',   # ratio of iou > 0.1
  'h_T_5r', 'h_T_10r', 'h_T_15r',   # ratio of iou/best_iou > 0.5
  'h_T_5i', 'h_T_10i', 'h_T_15i',   # average iou/best_iou
  'stop_5', 'stop_10', 'stop_15', 
  'ep_len_5', 'ep_len_10', 'ep_len_15',
]
nav_room_metric_names=[
  'd_0_5', 'd_0_10', 'd_0_15', 
  'r_T_5', 'r_T_10', 'r_T_15', 
  'r_e_5', 'r_e_10', 'r_e_15', 
  'stop_5', 'stop_10', 'stop_15', 
  'ep_len_5', 'ep_len_10', 'ep_len_15',
]
eqa_metric_names = [
  'acc_5', 'acc_10', 'acc_15',
  'ep_len_5', 'ep_len_10', 'ep_len_15',
]

# question types
qtypes = ['object_color_compare_inroom', 'object_color_compare_xroom', 
          'object_size_compare_inroom', 'object_size_compare_xroom', 
          'room_size_compare', 'object_dist_compare_inroom']


def compute_qtype_acc(predictions, qtype, key_name='vqa_ans'):
    """
    Compute QA accuracy for qtype
    """
    acc, cnt = 0, 0
    for entry in predictions:
        if key_name in entry and entry['type'] == qtype:
            if entry[key_name] == entry['answer']:
                acc += 1
            cnt += 1
    return acc / (cnt+1e-5), cnt

def nav_loc(eval_loader, h3d, model, 
            init_pos, init_nav_action,
            nav_id, nav_phrase, nav_type, nav_phrase_emb, 
            nav_state, loc_state,
            attr, max_episode_length,
            init_act_dist, stop_gate):
  """
  Inputs:
  - eval_loader
  - h3d
  - model
  - init_pos        : [x, y, z, yaw] 
  - init_nav_action : (1, ) int
  - nav_id  
  - nav_phrase
  - nav_type
  - nav_phrase_emb  : (1, 300) float
  - nav_state       : (1, rnn_size) float
  - loc_state       : (1, rnn_size) float
  - attr
  - max_episode_length
  - init_act_dist   : initial action distance to the target
  Return:
  - pred            : {nav_id, nav_phrase, nav_type, nav_actions, loc_actions, pos_queue, iou, is_inside_room}
  - img_feat        : one of {rnn_feat, ego_feat + rnn_feat, cube_feat}
  """
  # set target
  if nav_type == 'object': 
    h3d.set_target_object(h3d.objects[nav_id])
  else:
    h3d.set_target_room(h3d.rooms[nav_id])

  # initialize
  pred = {}
  pred['nav_id'] = nav_id
  pred['nav_phrase'] = nav_phrase
  pred['nav_type'] = nav_type

  h3d.env.reset(x=init_pos[0], y=init_pos[2], yaw=init_pos[3])
  init_dist_to_target = h3d.get_dist_to_target(h3d.env.cam.pos)
  dists_to_target = [init_dist_to_target]  # init_dist, done_dist
  pos_queue = [init_pos]
  episode_length = 0
  episode_done = True
  nav_actions = []
  loc_actions = []
  rewards = []

  # act
  ego_img = h3d.env.render()
  ego_img = (torch.from_numpy(ego_img.transpose(2,0,1)).float() / 255.).cuda()
  ego_feat = eval_loader.dataset.cnn.extract_feats(ego_img.unsqueeze(0), conv4_only=True)  # (1, 3200)
  collision = False
  rot_act, rot_act_cnt = None, 0
  nav_action = init_nav_action  # (1, )
  for step in range(max_episode_length):
    # forward model: nav_logprobs (1, 4), loc_logprobs (1, 2)
    episode_length += 1
    nav_logprobs, nav_state, loc_logprobs, loc_rnn_feat, loc_state = model.forward_step(
          ego_feat, nav_phrase_emb, nav_type, nav_action, nav_state, loc_state) 

    # we use controller to decide when to "STORE" instead
    # TODO: what if use both controller and navigator to decide when to stop together?
    if stop_gate == 'localizer':
      nav_logprobs[0][3] = -1e5  
    elif stop_gate == 'navigator':
      loc_logprobs[0][1] = -1e5

    # special case 1:
    # if prev. nav action is "forward" and collided and this action is again "forward"
    if nav_action.item() == 0 and collision and torch.exp(nav_logprobs[0]).argmax().item() == 0:
      nav_logprobs[0][0] = -1e5

    # special case 2:
    # if spinned around 6 times for same rotation
    if torch.exp(nav_logprobs[0]).argmax().item() == rot_act and rot_act_cnt > 5:
      nav_logprobs[0][torch.exp(nav_logprobs[0]).argmax().item()] = -1e5

    # sample nav_action and loc_action
    nav_action = torch.exp(nav_logprobs[0]).argmax().item()
    loc_action = torch.exp(loc_logprobs[0]).argmax().item()

    # accmulate rot_act
    if nav_action == 0: 
      rot_act = None
      rot_act_cnt = 0
    elif nav_action in [1,2]:
      if rot_act == nav_action:
        rot_act_cnt += 1
      else:
        rot_act = nav_action
        rot_act_cnt = 1

    # sample termination
    if stop_gate == 'localizer':
      if loc_action == 1:
        nav_action = 3  # navigator's stop comes from controller
    elif stop_gate == 'navigator':
      if nav_action == 3:
        loc_action = 1
    nav_actions += [nav_action]
    loc_actions += [loc_action]

    # interact with environment
    ego_img, _, episode_done, collision = h3d.step(nav_action)

    # prepare state for next action
    ego_img = (torch.from_numpy(ego_img.transpose(2,0,1)).float() / 255.).cuda()
    ego_feat = eval_loader.dataset.cnn.extract_feats(ego_img.unsqueeze(0), conv4_only=True)  # (1, 3200)
    nav_action = torch.LongTensor([nav_action]).cuda()  # (1, )

    # add to result
    dists_to_target.append(h3d.get_dist_to_target(h3d.env.cam.pos))
    pos_queue.append([h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                      h3d.env.cam.pos.z, h3d.env.cam.yaw])

    if episode_done:
      break

  # add to prediction
  pred['nav_actions'] = nav_actions
  pred['loc_actions'] = loc_actions
  pred['pos_queue'] = pos_queue
  if nav_type == 'object':
    pred['iou'] = h3d.compute_target_iou(nav_id)
    pred['best_iou'] = eval_loader.dataset.hid_tid_to_best_iou[h3d.env.house.house['id']+'_'+nav_id]
  else:
    pred['is_inside_room'] = h3d.is_inside_room(pos_queue[-1], h3d.rooms[nav_id])

  # extract img_feat
  img_feat = None
  if 'room_size' in attr and nav_type == 'room':
    cube_map = h3d.env.render_cube_map(mode='rgb')  # (h, wx6, 3)
    w = cube_map.shape[1] // 6
    assert cube_map.shape[1] / 6 == w
    imgs = []
    for i in range(4):  # we only wanna 4 panoroma images
      imgs.append(cube_map[:, i*w:(i+1)*w, :])
    cube_rgb = np.array(imgs, dtype=np.uint8)  # (4, h, w, c)
    cube_rgb = (torch.from_numpy(cube_rgb.transpose(0,3,1,2)).float() / 255.).cuda()
    cube_feat = eval_loader.dataset.cnn.extract_feats(cube_rgb, conv4_only=True)  # (4, 3200)
    img_feat = cube_feat.unsqueeze(0)  # (1, 4, 3200)
  else:
    if nav_type == 'object':
      if nav_action != 3:
        # we need to forward localizer one more step 
        # as current rnn_feat was on 2nd-last position 
        # Fine for nav_action == 3, as last position is same as 2nd-last one.
        localizer = model.object_localizer if nav_type == 'object' else model.room_localizer
        _, loc_rnn_feat, _ = localizer.forward_step(ego_feat, nav_phrase_emb, loc_state)  # (1, rnn_size)

      if 'dist' in attr:
        img_feat = torch.cat([ego_feat, loc_rnn_feat], 1)  # (1, 3200 + rnn_size)
      else:
        img_feat = loc_rnn_feat  # (1, rnn_size)

  # compute stats
  metrics_slug = {}
  metrics_slug['d_0_' + str(init_act_dist)] = dists_to_target[0]
  metrics_slug['ep_len_' + str(init_act_dist)] = episode_length
  metrics_slug['stop_' + str(init_act_dist)] = 1 if nav_action == 3 else 0
  if nav_type == 'object':
    metrics_slug['d_T_' + str(init_act_dist)] = dists_to_target[-1]
    metrics_slug['d_D_' + str(init_act_dist)] = dists_to_target[0] - dists_to_target[-1]
    metrics_slug['d_min_' + str(init_act_dist)] = float(np.array(dists_to_target).min())
    metrics_slug['h_T_' + str(init_act_dist) + 'a'] = 1 if pred['iou'] >= 0.1 else 0
    metrics_slug['h_T_' + str(init_act_dist) + 'r'] = 1 if pred['iou']/(pred['best_iou']+1e-5) >= 0.5 else 0
    metrics_slug['h_T_' + str(init_act_dist) + 'i'] = pred['iou']/(pred['best_iou'] + 1e-5)
  else:
    inside_room = []
    for p in pos_queue:
      inside_room.append(h3d.is_inside_room(p, h3d.rooms[nav_id]))
      if inside_room[-1] == True:
        metrics_slug['r_T_' + str(init_act_dist)] = 1
      else:
        metrics_slug['r_T_' + str(init_act_dist)] = 0
      if any([x == True for x in inside_room]) == True:
        metrics_slug['r_e_' + str(init_act_dist)] = 1
      else:
        metrics_slug['r_e_' + str(init_act_dist)] = 0

  return pred, img_feat, metrics_slug


def eval(args, split, dists):

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
    'gpu_id': 0,
    'max_threads_per_gpu': 1,
    'cfg': cfg,
    'to_cache': True,
    'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
    'map_resolution': 500,
    'pretrained_cnn_path': args.pretrained_cnn_path,
    'requires_imgs': True,
    'question_types': ['all'],
  }
  dataset = NavLocVqaRlDataset(**loader_kwargs)
  eval_loader = DataLoader(dataset, batch_size=1, num_workers=0)
  print('eval_loader set up.')

  # set up model
  checkpoint_path = '%s/%s.pth' % (args.checkpoint_dir, args.id)
  checkpoint = torch.load(checkpoint_path)
  opt = checkpoint['opt']
  model = ModelFactory(opt)
  model.load_state_dict(checkpoint['model_state'])
  model.eval()  # set eval mode!
  model.cuda()
  print('model loaded from %s.' % checkpoint_path)

  # load navigation module
  if args.use_rl_nav > 0:
    # room_navigator
    room_navigator_checkpoint = torch.load('pyutils/eqa_nav/output/nav_room/rl5.pth')
    room_navigator = Navigator(room_navigator_checkpoint['opt'])
    room_navigator.load_state_dict(room_navigator_checkpoint['model_state'])
    model.room_navigator = room_navigator
    model.room_navigator.eval().cuda()
    print('model.room_navigator updated.')
    # object_navigator
    object_navigator_checkpoint = torch.load('pyutils/eqa_nav/output/nav_object/rl5.pth')
    object_navigator = Navigator(object_navigator_checkpoint['opt'])
    object_navigator.load_state_dict(object_navigator_checkpoint['model_state'])
    model.object_navigator = object_navigator
    model.object_navigator.eval().cuda()
    print('model.object_navigator updated.')

  while True:
    # run evaluation for this epoch
    invalids = []
    num_done = 0

    # metrics
    nav_object_stats = {n: Statistics() for n in nav_object_metric_names}
    nav_room_stats = {n: Statistics() for n in nav_room_metric_names}
    eqa_stats = {n: Statistics() for n in eqa_metric_names}

    # reset envs
    eval_loader.dataset._load_envs(start_idx=0, in_order=True)

    # run
    done = False  # current epoch is not done yet
    predictions = []
    while not done:
      for batch in tqdm(eval_loader):
        # load qn data
        # batch = {idx, qid, house, question, answer, type, attr, qe, ae, path_name, path_ix
        # nav_ids, nav_types, nav_phrases, nav_phrase_embs, nav_ego_feats, nav_action_inputs/outputs}
        # private vars: episode_house (h3d), nav_pos_queues, path_len
        idx = batch['idx'][0].item()
        qid = batch['qid'][0].item()
        entry = {
          'qid': qid,
          'type': batch['type'][0],
          'house': batch['house'][0],
          'question': batch['question'][0],
          'answer': batch['answer'][0],
          'path_name': batch['path_name'][0],
          'path_ix': batch['path_ix'][0].item(),
          'attr': batch['attr'][0],
          'nav_ids': [_[0] for _ in batch['nav_ids']],
          'nav_types': [_[0] for _ in batch['nav_types']],
          'nav_phrases': [_[0] for _ in batch['nav_phrases']],
        }

        h3d = eval_loader.dataset.episode_house
        nav_phrases = batch['nav_phrases']
        nav_phrase_embs = batch['nav_phrase_embs'].cuda()    # (1, #navs, 300)
        nav_pos_queues = eval_loader.dataset.nav_pos_queues  # #navs of l [x, y, z, yaw]
        nav_ego_feats = batch['nav_ego_feats']               # #navs of (1, l, 3200)
        nav_action_inputs = batch['nav_action_inputs']       # #navs of (1, l)
        nav_types = entry['nav_types']    
        nav_ids = entry['nav_ids']
        attr = entry['attr']

        # evaluate at multiple initializations
        for dist in dists:

          # forward through lstm till spawn
          ref_nav_id = eval_loader.dataset.Questions[qid]['ref_nav_id']
          spawn_ix = nav_ids.index(ref_nav_id)
          spawn_nav_id = nav_ids[spawn_ix]
          spawn_pos_queue = nav_pos_queues[spawn_ix]  # list of l [x, y, z, yaw]
          spawn_ego_feats = nav_ego_feats[spawn_ix]   # (1, l, 3200) float
          spawn_action_inputs = nav_action_inputs[spawn_ix]  # (1, l) int64
          spawn_nav_type = nav_types[spawn_ix]
          spawn_phrase_emb = nav_phrase_embs[:, spawn_ix, :] # (1, 300)

          if len(spawn_pos_queue) < dist:
            # randomly spawn it 
            # note BFS distance maybe not strictly equal to #actions, but should be a good estimation.
            h3d.set_target_object(h3d.objects[spawn_nav_id]) if spawn_nav_type == 'object' else h3d.set_target_room(h3d.rooms[spawn_nav_id])
            init_pos, _ = eval_loader.dataset.spawn_agent(dist-2, dist, split)  # [x, y, z, yaw]
            if init_pos is None:  
              continue
            nav_state, loc_state = None, None
            init_nav_action = torch.LongTensor([eval_loader.dataset.act_to_ix['dummy']]).cuda()  # (1, ) 
          elif len(spawn_pos_queue) > dist:
            pos_queue = spawn_pos_queue[:-dist]  # till spawned position
            nav_state, loc_state = model.compute_states(
                                    spawn_nav_type, 
                                    spawn_ego_feats[:, :len(pos_queue)].cuda(),  # (1, spawn_dist, 3200)
                                    spawn_phrase_emb.cuda(),  # (1, 300)
                                    spawn_action_inputs[:, :len(pos_queue)].cuda(),  # (1, spawn_dist)
                                    )  # (1, rnn_size)
            init_nav_action = spawn_action_inputs[0, len(pos_queue)].view(-1).cuda() # (1, )
            init_pos = spawn_pos_queue[-dist]  # [x, y, z, yaw]
          else:
            nav_state, loc_state = None, None
            init_nav_action = torch.LongTensor([eval_loader.dataset.act_to_ix['dummy']]).cuda()  # (1, )
            init_pos = spawn_pos_queue[-dist]  # [x, y, z, yaw]

          # looking for targets one-by-one
          entry['d_%s'%dist] = []
          img_feats = []
          for i in range(len(nav_ids)):
            nav_id = nav_ids[i]
            nav_type = nav_types[i]
            nav_phrase = nav_phrases[i]
            nav_phrase_emb = nav_phrase_embs[:, i, :]  # (1, 300)
            max_episode_length = 50 if nav_type == 'object' else 120
            pred, img_feat, metrics_slug = nav_loc(eval_loader, h3d, model, init_pos, init_nav_action, 
                                                   nav_id, nav_phrase, nav_type, nav_phrase_emb, 
                                                   nav_state, loc_state, 
                                                   attr, max_episode_length, dist, args.stop_gate)

            # add to details and img_feats
            if img_feat is not None: 
              img_feats += [img_feat]
            entry['d_%s'%dist] += [pred]  # {nav_id, nav_phrase, nav_type, nav_actions, loc_actions, pos_queue, iou, is_inside_room}

            # update metrics
            if nav_type == 'object':
              for n in metrics_slug:
                nav_object_stats[n].push(metrics_slug[n])
            else:
              for n in metrics_slug:
                nav_room_stats[n].push(metrics_slug[n])

            # initialization for next nav
            init_pos = pred['pos_queue'][-1]
            init_nav_action = torch.LongTensor([eval_loader.dataset.act_to_ix['dummy']]).cuda()  # (1, )
            nav_state, loc_state = None, None

          # run vqa
          img_feats = torch.cat(img_feats, 0)  # (#tgts, feat_dim)
          eqa_ans_score = F.softmax(model.vqa([img_feats], [attr]), 1).view(-1)  # (#answers, )
          eqa_ans = eval_loader.dataset.itoa[eqa_ans_score.data.cpu().numpy().argmax()]
          entry['pred_ans_%s'%dist] = eqa_ans

          # print eqa
          total_ep_len = sum([len(pred['pos_queue']) for pred in entry['d_%s'%dist]])
          eqa_stats['acc_%s'%dist].push(1 if eqa_ans == entry['answer'] else 0)
          eqa_stats['ep_len_%s'%dist].push(total_ep_len)
          print('qid%s(%s/%s), total_ep_len: %s\n qn: %s\n gd_ans: %s, pred_ans_d%s: %s, acc_%s=%.2f%%.' % 
            (qid, num_done+1, len(eval_loader.dataset.questions), total_ep_len, entry['question'], 
            entry['answer'], dist, entry['pred_ans_%s'%dist], dist, eqa_stats['acc_%s'%dist].mean()*100.))

          metrics_slug = {'acc_%s'%dist: 1 if entry['pred_ans_%s'%dist] == entry['answer'] else 0}

        predictions.append(entry)
        num_done += 1

      # next environments
      eval_loader.dataset._load_envs()
      print("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
      if len(eval_loader.dataset.pruned_env_set) == 0:
        done = True

    # write summary
    if not args.use_rl_nav:
      results_str = 'id[%s] split[%s]\n' % (args.id, args.split)
    else:
      results_str = 'nav_rl+id[%s] split[%s] stop_gate[%s]\n' % (args.id, args.split, args.stop_gate)
    for metric_names, stats in zip([nav_object_metric_names, nav_room_metric_names, eqa_metric_names], 
                                     [nav_object_stats, nav_room_stats, eqa_stats]):
      for n in metric_names:
        if len(stats[n]) > 0:
          results_str += '%10s: %6.2f,' % (n, stats[n].mean())
          if '_15' in n: 
            results_str += '\n'
    for dist in dists:
      results_str += '  eqa-%ssteps: %.2f%%\n' % (dist, eqa_stats['acc_%s'%dist].mean()*100.)
      for qtype in qtypes:
        acc, cnt = compute_qtype_acc(predictions, qtype, 'pred_ans_%s'%dist)
        results_str += '  %-28s (%-3s): %.2f%%\n' % (qtype, cnt, acc*100.)
    f = open(args.report_txt, 'a')
    f.write(results_str)
    f.write('\n')
    f.close()

    # save results
    with open(args.result_json, 'w') as f:
      json.dump(predictions, f)
    print('predictions saved to %s.' % args.result_json)

    # break the outer [while True]
    break


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
  # Model settings
  parser.add_argument('--stop_gate', type=str, default='localizer')
  # Output settings
  parser.add_argument('--checkpoint_dir', type=str, default='output/reinforce')
  parser.add_argument('--id', type=str, default='im0')
  parser.add_argument('--use_rl_nav', type=int, default=1)
  parser.add_argument('--split', type=str, default='val')
  parser.add_argument('--path_ix', type=int, default=0)
  args = parser.parse_args()

  # make dirs
  if not osp.isdir('cache/reports/'): os.makedirs('cache/reports/')
  if not osp.isdir('cache/results/reinforce'): os.makedirs('cache/results/reinforce')
  args.report_txt = 'cache/reports/reinforce_random_spawned.txt'
  if not args.use_rl_nav:
    args.result_json = 'cache/results/reinforce/%s_%s_p%s_random_spawned.json' % (args.id, args.split, args.path_ix)
  else:
    if args.stop_gate == 'localizer':
      args.result_json = 'cache/results/reinforce/nav_rl_%s_%s_p%s_random_spawned.json' % (args.id, args.split, args.path_ix)
    elif args.stop_gate == 'navigator':
      args.result_json = 'cache/results/reinforce/nav_rl_%s_%s_p%s_random_spawned_stop_by_navigator.json' % (args.id, args.split, args.path_ix) 

  # run
  start = time.time()
  eval(args, args.split, [5, 10, 15])
  print('Evaluation [%s] done in %.2f seconds.' % (args.split, time.time() - start))
