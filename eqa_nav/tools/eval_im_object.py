import h5py
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
from nav.loaders.nav_imitation_loader import NavImitationDataset
from nav.models.crits import SeqModelCriterion, MaskedMSELoss
from nav.models.navigator import Navigator
import nav.models.utils as model_utils


def evaluate(val_dataset, model, nll_crit, mse_crit, opt):
  # set mode
  model.eval()

  # predict
  predictions = []
  overall_nll = 0
  overall_teacher_forcing_acc, overall_teacher_forcing_cnt = 0, 0
  overall_mse = 0
  Nav_nll = {'object': 0, 'room': 0}
  Nav_cnt = {'object': 0, 'room': 0}
  Nav_teacher_forcing_acc = {'object': 0, 'room': 0}
  Nav_teacher_forcing_cnt = {'object': 0, 'room': 0}
  for ix in range(len(val_dataset)):
    # data = {qid, path_ix, house, id, type, phrase, phrase_emb, ego_feats, next_feats, res_feats,
    #  action_inputs, action_outputs, action_masks, ego_imgs}
    data = val_dataset[ix]
    ego_feats = torch.from_numpy(data['ego_feats']).cuda().unsqueeze(0)  # (1, L, 3200)
    phrase_embs = torch.from_numpy(data['phrase_emb']).cuda().unsqueeze(0)  # (1, 300)
    action_inputs = torch.from_numpy(data['action_inputs']).cuda().unsqueeze(0)   # (1, L)
    action_outputs = torch.from_numpy(data['action_outputs']).cuda().unsqueeze(0) # (1, L)
    action_masks = torch.from_numpy(data['action_masks']).cuda().unsqueeze(0)  # (1, L)
    # forward
    logprobs, _, pred_feats, _ = model(ego_feats, phrase_embs, action_inputs)  # (1, L, #actions), (1, L, 3200)
    nll_loss = nll_crit(logprobs, action_outputs, action_masks)
    nll_loss = nll_loss.item()
    mse_loss = 0
    if opt['use_next']:
      next_feats = torch.from_numpy(data['next_feats']).cuda().unsqueeze(0)  # (1, L, 3200)
      mse_loss = mse_crit(pred_feats, next_feats, action_masks)
      mse_loss = mse_loss.item()
    if opt['use_residual']:
      res_feats = torch.from_numpy(data['res_feats']).cuda().unsqueeze(0)  # (1, L, 3200)
      mse_loss = mse_crit(pred_feats, res_feats, action_masks)
      mse_loss = mse_loss.item()
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
    entry['mse_loss'] = mse_loss
    # accumulate
    predictions.append(entry)
    Nav_nll[data['type']] += nll_loss
    Nav_cnt[data['type']] += 1
    acc, cnt = 0, 0
    for pa, ga in zip(entry['pred_acts'], entry['gd_acts']):
      if pa == ga:
        acc += 1
      cnt += 1
      if ga == 3:
        break
    Nav_teacher_forcing_acc[data['type']] += acc
    Nav_teacher_forcing_cnt[data['type']] += cnt
    overall_nll += nll_loss
    overall_mse += mse_loss
    overall_teacher_forcing_acc += acc
    overall_teacher_forcing_cnt += cnt
    # print
    if ix % 10 == 0:
      print('(%s/%s)qid[%s], id[%s], type[%s], nll_loss=%.3f, mse_loss=%.3f' % \
        (ix+1, len(val_dataset), entry['qid'], entry['id'], entry['type'], nll_loss, mse_loss))

  # summarize 
  overall_nll /= len(val_dataset)
  overall_mse /= len(val_dataset)
  overall_teacher_forcing_acc /= overall_teacher_forcing_cnt
  for _type in ['object', 'room']:
    Nav_nll[_type] /= (Nav_cnt[_type]+1e-5)
    Nav_teacher_forcing_acc[_type] /= (Nav_teacher_forcing_cnt[_type]+1e-5)
  
  # return
  return predictions, overall_nll, overall_teacher_forcing_acc, overall_mse, Nav_nll, Nav_teacher_forcing_acc, Nav_cnt

def main(args):

  # set up model
  checkpoint_path = osp.join(args.checkpoint_dir, '%s.pth' % args.id)
  checkpoint = torch.load(checkpoint_path)
  opt = checkpoint['opt']
  model = Navigator(checkpoint['opt'])
  model.load_state_dict(checkpoint['model_state'])
  model.cuda()
  print('model set up.')

  # set up criterions
  nll_crit = SeqModelCriterion().cuda()
  mse_crit = MaskedMSELoss().cuda()

  # set up loader
  loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': args.split,
    'max_seq_length': 100,
    'requires_imgs': False,
    'nav_types': ['object'],
    'question_types': ['all'],
  }
  dataset = NavImitationDataset(**loader_kwargs)

  # evaluate 
  predictions, overall_nll, overall_teacher_forcing_acc, overall_mse, Nav_nll, Nav_teacher_forcing_acc, Nav_cnt = \
        evaluate(dataset, model, nll_crit, mse_crit, opt)

  # summarize
  results_str = 'id[%s] ' % args.id
  if opt['use_action']: results_str += '[use action]'
  if opt['use_curriculum']: results_str += '[use curriculum]'
  if opt['use_next']: results_str += '[use next]'
  if opt['use_residual']: results_str += '[use residual]'
  results_str += '\nsplit[%s]\n' % args.split

  results_str += '  nll_loss: %.3f\n' % overall_nll
  results_str += '  teacher-forcing acc (%s): %.2f%%,' % (len(predictions), overall_teacher_forcing_acc * 100.)
  results_str += ' on %s objects: %.2f%%,' % (Nav_cnt['object'], Nav_teacher_forcing_acc['object']*100.)
  results_str += ' on %s rooms: %.2f%%\n' % (Nav_cnt['room'], Nav_teacher_forcing_acc['room']*100.)

  # save
  with open(args.result_json, 'w') as f:
    json.dump(predictions, f)
  f = open(args.report_txt, 'a')
  f.write(results_str)
  f.write('\n')
  f.close()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  # Data input settings
  parser.add_argument('--data_json', type=str, default='cache/prepro/imitation/data.json')
  parser.add_argument('--data_h5', type=str, default='cache/prepro/imitation/data.h5')
  parser.add_argument('--path_feats_dir', type=str, default='cache/path_feats')
  parser.add_argument('--path_images_dir', type=str, default='cache/path_images')
  parser.add_argument('--checkpoint_dir', type=str, default='output/nav_object')
  # Output settings
  parser.add_argument('--id', type=str, default='im0')
  parser.add_argument('--split', type=str, default='val')
  args = parser.parse_args()

  # make dirs
  if not osp.isdir('cache/reports'): os.makedirs('cache/reports')
  args.report_txt = 'cache/reports/nav_object_ll.txt'
  if not osp.isdir('cache/results/nav_object'): os.makedirs('cache/results/nav_object')
  args.result_json = 'cache/results/nav_object/%s_%s_ll.json' % (args.id, args.split)

  # run
  main(args)
