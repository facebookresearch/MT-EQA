# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import csv
import time
import argparse
import random
import numpy as np
import os, sys, json
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import _init_paths
from nav_loc_vqa.models.crits import SeqModelCriterion
from nav_loc_vqa.loaders.nav_im_loader import NavImitationDataset
from nav_loc_vqa.loaders.nav_loc_vqa_im_loader import NavLocVqaImDataset 
from nav_loc_vqa.models.factory import ModelFactory 
import nav_loc_vqa.models.utils as model_utils

import tensorboardX as tb
from datetime import datetime


def get_semantic_classes(semantic_color_file):
  """
  return fine_class -> ix, rgb
  - cls_to_ix, cls_to_rgb 
  """
  cls_to_ix, cls_to_rgb = {}, {}
  with open(semantic_color_file) as csv_file:
    reader = csv.DictReader(csv_file)
    for i, row in enumerate(reader):
      fine_cat = row['name']
      cls_to_rgb[fine_cat] = (row['r'], row['g'], row['b'])
      cls_to_ix[fine_cat] = i
  return cls_to_ix, cls_to_rgb 

def evaluate_nav(val_dataset, model, nav_crit):
  # set mode
  model.eval()

  # predict
  predictions = []
  nav_nll = {'object': 0, 'room': 0}
  tf_acc = {'object': 0, 'room': 0}
  tf_cnt = {'object': 0, 'room': 0}
  for ix in range(len(val_dataset)):
    # data = {qid, path_ix, house, id, type, phrase, phrase_emb, ego_feats, next_feats, res_feats,
    #  action_inputs, action_outputs, action_masks, ego_imgs}
    data = val_dataset[ix]
    ego_feats = torch.from_numpy(data['ego_feats']).cuda().unsqueeze(0)  # (1, L, 3200)
    phrase_embs = torch.from_numpy(data['phrase_emb']).cuda().unsqueeze(0)  # (1, 300)
    action_inputs = torch.from_numpy(data['action_inputs']).cuda().unsqueeze(0)   # (1, L)
    action_outputs = torch.from_numpy(data['action_outputs']).cuda().unsqueeze(0) # (1, L)
    action_masks = torch.from_numpy(data['action_masks']).cuda().unsqueeze(0)  # (1, L)
    # forward -> (1, L, #actions)
    if data['type'] == 'object':
      logprobs, _, _, _ = model.object_navigator(ego_feats, phrase_embs, action_inputs)   
    else:
      logprobs, _, _, _ = model.room_navigator(ego_feats, phrase_embs, action_inputs)
    nll_loss = nav_crit(logprobs, action_outputs, action_masks)
    nll_loss = nll_loss.item()
    pred_acts = logprobs[0].argmax(1)  # (L, )
    # entry
    entry = {}
    entry['qid'] = data['qid']
    entry['house'] = data['house']
    entry['id'] = data['id']
    entry['type'] = data['type']
    entry['path_ix'] = data['path_ix']
    entry['pred_acts'] = pred_acts.tolist()        # list of L actions
    entry['pred_acts_probs'] = torch.exp(logprobs[0]).tolist() # (L, #actions)
    entry['gd_acts'] = action_outputs[0].tolist()  # list of L actions
    entry['nll_loss'] = nll_loss
    # accumulate
    predictions.append(entry)
    acc, cnt = 0, 0
    for pa, ga in zip(entry['pred_acts'], entry['gd_acts']):
      if pa == ga:
        acc += 1
      cnt += 1
      if ga == 3:
        break
    tf_acc[data['type']] += acc
    tf_cnt[data['type']] += cnt
    nav_nll[data['type']] += nll_loss
    # print
    if ix % 10 == 0:
      print('(%s/%s)qid[%s], id[%s], type[%s], nll_loss=%.3f' % \
        (ix+1, len(val_dataset), entry['qid'], entry['id'], entry['type'], nll_loss))

  # summarize
  for _type in ['object', 'room']:
    tf_acc[_type] /= tf_cnt[_type]
    nav_nll[_type] /= tf_cnt[_type]
  
  # return
  return predictions, nav_nll['room'], tf_acc['room'], nav_nll['object'], tf_acc['object']

def evaluate(val_dataset, model, path_ix, cls_to_rgb):
  # set mode
  model.eval()

  # predict
  predictions = []
  vqa_acc = 0
  eqa_acc = 0
  loc_object_acc, num_loc_object = 0, 0
  loc_room_acc, num_loc_room = 0, 0
  for i, qid in enumerate(val_dataset.ids):
    # get test data = {qid, questions, answer, type, attr, qe, ae, path_ix, path_len,
    # Boxes, key_ixs, tgt_ids,
    # nav_ids, nav_phrases, nav_phrase_embs, nav_types
    # ego_feats, phrase_embs, phrases, 
    # cube_feats, room_ids, room_phrases, room_phrase_embs, room_to_inroomDists, room_to_labels}
    data = val_dataset.getTestData(qid, path_ix=path_ix, requires_imgs=False)
    ego_feats = torch.from_numpy(data['ego_feats']).cuda()   # (L, 3200)
    cube_feats = torch.from_numpy(data['cube_feats']).cuda() # (L, 4, 3200)
    attr = data['attr']
    nav_types = data['nav_types']

    # forward with key_ixs -> vqa_ans_sc (#answers, )
    tgt_key_ixs = torch.from_numpy(np.array(data['tgt_key_ixs'], dtype=np.int64)).cuda()  # (#tgt, )
    tgt_phrase_embs = torch.from_numpy(data['tgt_phrase_embs']).cuda()  # (#tgt, 300)
    vqa_ans_score = model.gd_path_key_to_answer(ego_feats, cube_feats, tgt_phrase_embs, tgt_key_ixs, attr)

    # forward with nothing -> loc_probs (#nav, L), sample_ixs (#nav, ), ans_sc (#answers, )
    nav_phrase_embs = torch.from_numpy(data['nav_phrase_embs']).cuda()  # (#nav, 300)
    nav_types = data['nav_types']
    loc_probs, sample_ixs, eqa_ans_score = model.gd_path_to_sample_answer(ego_feats, cube_feats, nav_phrase_embs, nav_types, attr)

    # navigation's meta: [{nav_id, sample_ix, type, iou/inroomDist}]
    pred_navs = []
    for nav_id, nav_type, sample_ix in zip(data['nav_ids'], data['nav_types'], sample_ixs.tolist()):
      pred_nav = {}
      pred_nav['nav_id'] = nav_id
      pred_nav['type'] = nav_type
      pred_nav['sample_ix'] = sample_ix
      if nav_type == 'object':
        ego_sem = data['ego_sems'][sample_ix]  # (224, 224, 3)
        fine_class = data['Boxes'][nav_id]['fine_class']
        pred_nav['iou'] = model_utils.compute_obj_iou(ego_sem, cls_to_rgb, fine_class)
      else:
        pred_nav['inroomDist'] = data['room_to_inroomDists'][nav_id][sample_ix]
      pred_navs.append(pred_nav)

    # entry
    entry = {}
    entry['qid'] = data['qid']
    entry['question'] = data['question']
    entry['answer'] = data['answer']
    entry['type'] = data['type']
    entry['path_ix'] = data['path_ix']
    entry['attr'] = data['attr']
    # --
    entry['nav_phrases'] = data['nav_phrases']
    entry['nav_types'] = data['nav_types']
    entry['nav_ids'] = data['nav_ids']
    entry['loc_probs'] = loc_probs.tolist()      # (#nav, L)
    entry['sample_ixs'] = sample_ixs.tolist()    # (#nav,)
    entry['eqa_ans_score'] = eqa_ans_score.tolist() # (#answers, )
    entry['eqa_ans'] = val_dataset.itoa[eqa_ans_score.data.cpu().numpy().argmax()]
    # --
    entry['tgt_ids'] = data['tgt_ids']
    entry['tgt_key_ixs'] = data['tgt_key_ixs']  # (#tgt, )
    entry['vqa_ans_score'] = vqa_ans_score.tolist() # (#answers, )
    entry['vqa_ans'] = val_dataset.itoa[vqa_ans_score.data.cpu().numpy().argmax()]
    # --
    entry['pred_navs'] = pred_navs # [{nav_id, sample_ix, type, iou/inroomDist}]

    # accumulate
    predictions.append(entry)
    if entry['vqa_ans'] == entry['answer']: vqa_acc += 1
    if entry['eqa_ans'] == entry['answer']: eqa_acc += 1
    for pred_nav in pred_navs:
      if pred_nav['type'] == 'object': 
        if pred_nav['iou'] >= 0.1: loc_object_acc += 1
        num_loc_object += 1
      if pred_nav['type'] == 'room': 
        if pred_nav['inroomDist'] > 0: loc_room_acc += 1
        num_loc_room += 1

    # print
    if i % 10 == 0:
      print('(%s/%s)qid[%s], tgt_key_ixs=%s, sample_ixs=%s, gd_ans=%s, eqa_ans=%s.' % \
        (i, len(val_dataset.ids), entry['qid'], entry['tgt_key_ixs'], entry['sample_ixs'], 
        entry['answer'], entry['eqa_ans']))

  # summarize
  vqa_acc /= len(val_dataset.ids)
  eqa_acc /= len(val_dataset.ids)
  loc_object_acc /= num_loc_object
  loc_room_acc /= num_loc_room
  print('Evaluating [%s] %s localized_eqa, loc_object_acc=%.2f%%, loc_room_acc=%.2f%%, vqa_acc=%.2f%%, eqa_acc=%.2f%%' % \
    ('val', len(val_dataset.ids), loc_object_acc*100., loc_room_acc*100., vqa_acc*100., eqa_acc*100.))
  # return
  return predictions, vqa_acc, eqa_acc, loc_object_acc, loc_room_acc

def main(args):
  # make output directory
  if not osp.isdir(args.checkpoint_dir): 
    os.makedirs(args.checkpoint_dir)  

  # tensorboard logdir
  if not osp.isdir(args.tb_dir): 
    os.makedirs(args.tb_dir)
  now = datetime.now()
  writer = tb.SummaryWriter(osp.join(args.tb_dir, args.id, now.strftime("%Y%m%d-%H%M")))

  # set random seed
  random.seed(args.seed)
  np.random.randn(args.seed)
  torch.manual_seed(args.seed)
  # torch.cuda.manual_seed(args.seed)
  # torch.backend.cudnn.deterministic = True

  # set up fine_class --> ix/rgb
  color_file = osp.join(args.house3d_metadata_dir, 'colormap_fine.csv')
  cls_to_ix, cls_to_rgb = get_semantic_classes(color_file)

  # set up loaders
  train_loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'train',
    'room_seq_length': args.room_seq_length,
    'object_seq_length': args.object_seq_length,
  }
  train_dataset = NavLocVqaImDataset(**train_loader_kwargs)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
  val_loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'val',
    'room_seq_length': args.room_seq_length,
    'object_seq_length': args.object_seq_length,
  }
  val_dataset = NavLocVqaImDataset(**val_loader_kwargs)
  val_nav_loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'val',
    'max_seq_length': max(args.object_seq_length, args.room_seq_length),
    'requires_imgs': False,
  }
  val_nav_dataset = NavImitationDataset(**val_nav_loader_kwargs)

  # set up models
  opt = vars(args)
  opt['atoi'] = train_dataset.atoi
  model = ModelFactory(opt)
  model.cuda()
  print('Model Factory set up.')

  # set up criterions
  nav_crit = SeqModelCriterion().cuda()
  loc_crit = SeqModelCriterion().cuda()
  vqa_crit = nn.CrossEntropyLoss().cuda()

  # set up optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                betas=(args.optim_alpha, args.optim_beta), eps=args.optim_epsilon,
                weight_decay=args.weight_decay)

  # resume from checkpoint
  infos = {}
  iters = infos.get('iters', 0)
  epoch = infos.get('epoch', 0)
  # validation log
  val_vqa_acc_history = infos.get('val_vqa_acc_history', {})
  val_eqa_acc_history = infos.get('val_eqa_acc_history', {})
  val_loc_room_acc_history = infos.get('val_loc_room_acc_history', {})
  val_loc_object_acc_history = infos.get('val_loc_object_acc_history', {})
  val_nav_room_nll_history = infos.get('val_nav_room_nll_history', {})
  val_nav_object_nll_history = infos.get('val_nav_object_nll_history', {})
  val_nav_room_tf_acc_history = infos.get('val_nav_room_tf_acc_history', {})
  val_nav_object_tf_acc_history = infos.get('val_nav_object_tf_acc_history', {})
  # training log
  loss_history = infos.get('loss_history', {})
  vqa_loss_history = infos.get('val_loss_history', {})
  loc_object_loss_history = infos.get('loc_object_loss_history', {})
  loc_room_loss_history = infos.get('loc_room_loss_history', {})
  nav_object_loss_history = infos.get('nav_object_loss_history', {})
  nav_room_loss_history = infos.get('nav_room_loss_history', {})
  lr = infos.get('lr', args.learning_rate)
  best_val_score, best_predictions, best_nav_predictions = None, None, None

  # start training
  while iters <= args.max_iters:
    print('Starting epoch %d' % epoch)
    for batch in train_loader:
      # set mode
      model.train()
      # zero gradient
      optimizer.zero_grad()
      # batch = {qid, question, answer, attr, qe, ae
      # object_ego_feats, object_phrases, object_phrase_embs, object_key_ixs, object_masks, object_labels,
      # object_action_inputs, object_action_outputs, object_actionmasks
      # room_ego_feats, room_phrases, room_phrase_embs, room_cubea_feats, room_key_ixs, room_select_ixs,
      # room_masks, room_labels
      # room_action_inputs, room_action_outputs, room_action_masks
      qids = batch['qid'].tolist()
      attrs = batch['attr']
      ae = batch['ae'].long().cuda()        # (n, ) long
      assert 0 <= ae.min().item() and ae.max().item() <= 1, ae
      # object-related
      object_ego_feats = batch['object_ego_feats'].cuda() # (n, 3, Lo, 3200) float
      object_labels = batch['object_labels'].cuda()       # (n, 3, Lo) long
      object_masks = batch['object_masks'].cuda()         # (n, 3, Lo) float
      object_phrase_embs = batch['object_phrase_embs'].cuda() # (n, 3, 300) float
      object_select_ixs = batch['object_key_ixs'].cuda()  # (n, 3) long
      object_action_inputs = batch['object_action_inputs'].cuda()   # (n, 3, Lo) long
      object_action_outputs = batch['object_action_outputs'].cuda() # (n, 3, Lo) long
      object_action_masks = batch['object_action_masks'].cuda()     # (n, 3, Lo) float
      # room-related
      room_ego_feats = batch['room_ego_feats'].cuda()  # (n, 2, Lr, 3200) float
      room_labels = batch['room_labels'].cuda()        # (n, 2, Lr) long
      room_masks = batch['room_masks'].cuda()          # (n, 2, Lr) float
      room_phrase_embs = batch['room_phrase_embs'].cuda() # (n, 2, 300) float
      room_select_ixs = batch['room_select_ixs'].cuda()   # (n, 2,) long
      room_cube_feats = batch['room_cube_feats'].cuda()   # (n, 2, 4, 3200) float
      room_action_inputs = batch['room_action_inputs'].cuda()   # (n, 2, Lr) long
      room_action_outputs = batch['room_action_outputs'].cuda() # (n, 2, Lr) long
      room_action_masks = batch['room_action_masks'].cuda()     # (n, 2, Lr) long
      # forward model
      room_action_logprobs, object_action_logprobs, room_loc_logprobs, object_loc_logprobs, scores = \
        model(object_ego_feats, object_phrase_embs, object_masks, object_select_ixs, object_action_inputs,
              room_ego_feats, room_phrase_embs, room_masks, room_select_ixs, room_cube_feats,
              room_action_inputs, attrs)
      # loss 
      nav_room_loss = nav_crit(room_action_logprobs, room_action_outputs.view(len(qids)*2, -1), room_action_masks.view(len(qids)*2, -1))
      nav_object_loss = nav_crit(object_action_logprobs, object_action_outputs.view(len(qids)*3, -1), object_action_masks.view(len(qids)*3, -1))
      loc_room_loss = loc_crit(room_loc_logprobs, room_labels.view(len(qids)*2, -1), room_masks.view(len(qids)*2, -1))
      loc_object_loss = loc_crit(object_loc_logprobs, object_labels.view(len(qids)*3, -1), object_masks.view(len(qids)*3, -1))
      vqa_loss = vqa_crit(scores, ae)
      loss = args.nav_room_weight * nav_room_loss + args.nav_object_weight * nav_object_loss + \
             args.loc_room_weight * loc_room_loss + args.loc_object_weight * loc_object_loss + \
             args.vqa_weight * vqa_loss
      # backward
      loss.backward()
      model_utils.clip_gradient(optimizer, args.grad_clip)
      optimizer.step()

      # training log
      if iters % args.losses_log_every == 0:
        loss_history[iters] = loss.item()
        nav_room_loss_history[iters] = nav_room_loss.item()
        nav_object_loss_history[iters] = nav_object_loss.item()
        loc_room_loss_history[iters] = loc_room_loss.item()
        loc_object_loss_history[iters] = loc_object_loss.item()
        vqa_loss_history[iters] = vqa_loss.item()
        print('iters[%s]epoch[%s], train_loss=%.3f (nav_room=%.3f, nav_obj=%.3f, loc_room=%.3f, loc_obj=%.3f, vqa_loss=%.3f) lr=%.2E' % \
          (iters, epoch, loss.item(), nav_room_loss.item(), nav_object_loss.item(), loc_room_loss.item(), loc_object_loss.item(), vqa_loss.item(), lr))
        writer.add_scalar('train_loss', loss.item(), iters)
        writer.add_scalar('train_nav_room_loss', nav_room_loss.item(), iters)
        writer.add_scalar('train_nav_object_loss', nav_object_loss.item(), iters)
        writer.add_scalar('train_loc_room_loss', loc_room_loss.item(), iters)
        writer.add_scalar('train_loc_object_loss', loc_object_loss.item(), iters)
        writer.add_scalar('train_vqa_loss', vqa_loss.item(), iters)
        
      # decay learning rate
      if args.learning_rate_decay_start > 0 and iters > args.learning_rate_decay_start:
        frac = (iters - args.learning_rate_decay_start) / args.learning_rate_decay_every
        decay_factor = 0.1 ** frac
        lr = args.learning_rate * decay_factor
        model_utils.set_lr(optimizer, lr)

      # evaluate
      if iters % args.save_checkpoint_every == 0:

        print('Checking validation ...')
        predictions, vqa_acc, eqa_acc, loc_room_acc, loc_object_acc = evaluate(val_dataset, model, 0, cls_to_rgb)
        nav_predictions, nav_room_nll, nav_room_tf_acc, nav_object_nll, nav_object_tf_acc = evaluate_nav(val_nav_dataset, model, nav_crit)
        val_vqa_acc_history[iters] = vqa_acc
        val_eqa_acc_history[iters] = eqa_acc
        val_loc_room_acc_history[iters] = loc_room_acc
        val_loc_object_acc_history[iters] = loc_object_acc
        val_nav_room_nll_history[iters] = nav_room_nll
        val_nav_room_tf_acc_history[iters] = nav_room_tf_acc
        val_nav_object_nll_history[iters] = nav_object_nll
        val_nav_object_tf_acc_history[iters] = nav_object_tf_acc
        writer.add_scalar('val_vqa_acc', vqa_acc, iters)
        writer.add_scalar('val_eqa_acc', eqa_acc, iters)
        writer.add_scalar('val_loc_room_acc', loc_room_acc, iters)
        writer.add_scalar('val_loc_object_acc', loc_object_acc, iters)
        writer.add_scalar('val_nav_room_nll', nav_room_nll, iters)
        writer.add_scalar('val_nav_room_tf_acc', nav_room_tf_acc, iters)
        writer.add_scalar('val_nav_object_nll', nav_object_nll, iters)
        writer.add_scalar('val_nav_object_tf_acc', nav_object_tf_acc, iters)

        # save model if best
        # empirically consider all three accuracies; perhaps a better weighting is needed.
        current_score = eqa_acc + loc_object_acc + loc_room_acc   
        if best_val_score is None or current_score > best_val_score:
          best_val_score = current_score
          best_predictions = predictions
          checkpoint_path = osp.join(args.checkpoint_dir, '%s.pth' % args.id)
          checkpoint = {}
          checkpoint['model_state'] = model.state_dict()
          checkpoint['opt'] = vars(args)
          torch.save(checkpoint, checkpoint_path)
          print('model saved to %s.' % checkpoint_path)
        
        # write to json report
        infos['iters'] = iters
        infos['epoch'] = epoch
        infos['loss_history'] = loss_history
        infos['loc_room_loss_history'] = loc_room_loss_history
        infos['loc_object_loss_history'] = loc_object_loss_history
        infos['vqa_loss_history'] = vqa_loss_history
        infos['val_vqa_acc_history'] = val_vqa_acc_history
        infos['val_eqa_acc_history'] = val_eqa_acc_history
        infos['val_loc_room_acc_history'] = val_loc_room_acc_history
        infos['val_loc_object_acc_history'] = val_loc_object_acc_history
        infos['val_nav_room_nll_history'] = val_nav_room_nll_history
        infos['val_nav_room_tf_acc_history'] = val_nav_room_tf_acc_history
        infos['val_nav_object_nll_history'] = val_nav_object_nll_history
        infos['val_nav_object_tf_acc_history'] = val_nav_object_tf_acc_history
        infos['best_val_score'] = best_val_score
        infos['best_predictions'] = predictions if best_predictions is None else best_predictions
        infos['best_nav_predictions'] = nav_predictions if best_nav_predictions is None else best_nav_predictions
        infos['opt'] = vars(args)
        infos['wtov'] = train_dataset.wtov
        infos_json = osp.join(args.checkpoint_dir, '%s.json' % args.id)
        with open(infos_json, 'w') as f:
          json.dump(infos, f)
        print('infos saved to %s.' % infos_json)

      # update iters
      iters += 1

    # update epoch
    epoch += 1


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  # Data input settings
  parser.add_argument('--data_json', type=str, default='cache/prepro/imitation/data.json')
  parser.add_argument('--data_h5', type=str, default='cache/prepro/imitation/data.h5')
  parser.add_argument('--path_feats_dir', type=str, default='cache/path_feats')
  parser.add_argument('--path_images_dir', type=str, default='cache/path_images')
  parser.add_argument('--checkpoint_dir', type=str, default='output/imitation')
  parser.add_argument('--tb_dir', type=str, default='logs/imitation')
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--house3d_metadata_dir', type=str, default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--seed', type=int, default=24)
  parser.add_argument('--start_from', type=str, default=None)  
  # Localizer settings
  parser.add_argument('--object_seq_length', type=int, default=50, help='seq_length on object')
  parser.add_argument('--room_seq_length', type=int, default=100, help='seq_length on room')
  parser.add_argument('--rnn_type', type=str, default='lstm')
  parser.add_argument('--rnn_size', type=int, default=256)
  parser.add_argument('--rnn_fc_dim', type=int, default=64, help='fc size for phrase and visual features')
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--rnn_dropout', type=float, default=0.1, help='dropout between rnn layer')
  parser.add_argument('--seq_dropout', type=float, default=0., help='dropout at rnn output')
  parser.add_argument('--loc_object_weight', type=float, default=2.0, help='localization weight on object')
  parser.add_argument('--loc_room_weight', type=float, default=2.0, help='localization weight on object')
  # Navigator settings (rnn setting same as localizer, just too lazy...)
  parser.add_argument('--nav_object_weight', type=float, default=1.0, help='navigation weight on object')
  parser.add_argument('--nav_room_weight', type=float, default=1.0, help='navigation weight on room')
  # VQA settings
  parser.add_argument('--vqa_weight', type=float, default=1.0)
  parser.add_argument('--qn_fc_dim', type=int, default=64)
  parser.add_argument('--qn_fc_dropout', type=float, default=0.)
  # Output settings
  parser.add_argument('--id', type=str, default='checkpoint0')
  parser.add_argument('--save_checkpoint_every', type=str, default=2000, help='how often to save a model checkpoint')
  parser.add_argument('--losses_log_every', type=int, default=25)
  # Optimizer
  parser.add_argument('--max_iters', type=int, default=20000, help='max number of iterations to run')
  parser.add_argument('--batch_size', type=int, default=20, help='batch size in number of questions per batch')
  parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
  parser.add_argument('--learning_rate_decay_start', type=int, default=5000, help='at what iters to start decaying learning rate')
  parser.add_argument('--learning_rate_decay_every', type=int, default=5000, help='every how many iters thereafter to drop LR by half')
  parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
  parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
  parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
  parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for l2 regularization')

  args = parser.parse_args()
  main(args)
