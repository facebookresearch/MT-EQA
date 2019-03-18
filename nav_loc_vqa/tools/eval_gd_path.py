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

qtypes = ['object_color_compare_inroom', 'object_color_compare_xroom', 
          'object_size_compare_inroom', 'object_size_compare_xroom', 
          'room_size_compare', 'object_dist_compare_inroom']

def compute_qtype_acc(predictions, qtype, key_name='vqa_ans'):
  """
  Compute QA accuracy for qtype
  """
  acc, cnt = 0, 0
  for entry in predictions:
    if entry['type'] == qtype:
      if entry[key_name] == entry['answer']:
        acc += 1
      cnt += 1
  return acc / (cnt+1e-5), cnt

# fine_class -> ix, rgb
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

  # set up model
  checkpoint_path = '%s/%s.pth' % (args.checkpoint_dir, args.id)
  checkpoint = torch.load(checkpoint_path)
  opt = checkpoint['opt']
  model = ModelFactory(opt)
  model.load_state_dict(checkpoint['model_state'])
  model.cuda()
  print('model factory set up.')

  nav_crit = SeqModelCriterion().cuda()
  print('nav_crit set up.')

  # set up fine_class -> ix/rgb
  color_file = osp.join(args.house3d_metadata_dir, 'colormap_fine.csv')
  cls_to_ix, cls_to_rgb = get_semantic_classes(color_file)

  # set up loader
  loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': args.split,
    'room_seq_length': opt['room_seq_length'],
    'object_seq_length': opt['object_seq_length'],
  }
  dataset= NavLocVqaImDataset(**loader_kwargs)
  nav_loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': args.split,
    'max_seq_length': max(opt['room_seq_length'], opt['object_seq_length']),
    'requires_imgs': False,
  }
  nav_dataset = NavImitationDataset(**nav_loader_kwargs)

  # evaluate overall vqa, eqa, loc_obj, loc_room
  predictions, vqa_acc, eqa_acc, loc_room_acc, loc_object_acc = evaluate(dataset, model, args.path_ix, cls_to_rgb)
  nav_predictions, nav_room_nll, nav_room_tf_acc, nav_object_nll, nav_object_tf_acc = evaluate_nav(nav_dataset, model, nav_crit)

  results_str = 'id[%s], split[%s], path_ix[%s]\n' % (args.id, args.split, args.path_ix)
  results_str += 'nav_room_tf_acc=%.2f%%, nav_object_tf_acc=%.2f%%\n' % (nav_room_tf_acc*100., nav_object_tf_acc*100.)
  results_str += 'loc_room_acc=%.2f%%, loc_object_acc=%.2f%%\n' % (loc_room_acc*100., loc_object_acc*100.)

  ############# OLD METRICS #############
  # evaluate loc_obj using old metrics (window-based f1, iou, dist, miss)
  object_predictions = [entry for entry in predictions if 'object' in entry['type']]
  for entry in object_predictions:
    entry['sampled_object_ixs'] = []
    entry['object_seq_probs'] = []
    for i, pred_nav in enumerate(entry['pred_navs']):
      if pred_nav['type'] == 'object':
        entry['sampled_object_ixs'].append(pred_nav['sample_ix'])
        entry['object_seq_probs'] += [entry['loc_probs'][i]]
    assert len(entry['sampled_object_ixs']) == len(entry['tgt_key_ixs'])
    assert len(entry['object_seq_probs']) == len(entry['tgt_key_ixs'])
  precision, recall, f1 = model_utils.evaluate_precision_recall_fscore(object_predictions, 'tgt_key_ixs', 'sampled_object_ixs', 5)
  iou, dist = model_utils.evaluate_distance_iou(object_predictions, 'tgt_key_ixs', 'sampled_object_ixs', 5)
  miss = model_utils.evaluate_miss_rate(object_predictions, 'object_seq_probs')
  results_str += 'hit on %s objects: p=%.2f, r=%.2f, f1=%.2f, iou=%.3f, dist=%.3f, miss=%.2f%%\n' % \
    (len(object_predictions), precision*100., recall*100., f1*100., iou, dist, miss*100.)
  
  # evaluate loc_room using old metrics
  room_predictions = [entry for entry in predictions if entry['type'] == 'room_size_compare']
  room_acc, room_cnt = 0, 0
  for entry in room_predictions:
    entry['sampled_room_ixs'] = []
    entry['room_seq_probs'] = []
    for i, pred_nav in enumerate(entry['pred_navs']):
      if pred_nav['type'] == 'room':
        entry['sampled_room_ixs'].append(pred_nav['sample_ix'])
        entry['room_seq_probs'] += [entry['loc_probs'][i]]
      if pred_nav['inroomDist'] > 0:
        room_acc += 1
      room_cnt +=1
    assert len(entry['room_seq_probs']) == len(entry['tgt_key_ixs'])
  precision, recall, f1 = model_utils.evaluate_precision_recall_fscore(room_predictions, 'tgt_key_ixs', 'sampled_room_ixs', 5)
  iou, dist = model_utils.evaluate_distance_iou(room_predictions, 'tgt_key_ixs', 'sampled_room_ixs', 5)
  miss = model_utils.evaluate_miss_rate(room_predictions, 'room_seq_probs')
  room_acc /= room_cnt 
  results_str += 'hit on %s rooms: p=%.2f, r=%.2f, f1=%.2f, iou=%.3f, dist=%.3f, miss=%.2f%%, inroom-acc=%.2f%%\n' % \
    (len(room_predictions), precision*100., recall*100., f1*100., iou, dist, miss*100., room_acc*100.)
  
  # decompose predictions into question type and report accuracy
  results_str += 'vqa accuracy: %.2f%%\n' % (vqa_acc*100.)
  for qtype in qtypes:
    acc, cnt = compute_qtype_acc(predictions, qtype, 'vqa_ans')
    results_str += '%-28s (%-3s): %.2f%%\n' % (qtype, cnt, acc*100.)
  
  results_str += 'eqa accuracy: %.2f%%\n' % (eqa_acc*100.)
  for qtype in qtypes:
    acc, cnt = compute_qtype_acc(predictions, qtype, 'eqa_ans')
    results_str += '%-28s (%-3s): %.2f%%\n' % (qtype, cnt, acc*100.)

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
  parser.add_argument('--house3d_metadata_dir', type=str, default='pyutils/House3D/House3D/metadata')
  # Output settings
  parser.add_argument('--checkpoint_dir', type=str, default='output/imitation')
  parser.add_argument('--id', type=str, default='checkpoint0')
  parser.add_argument('--split', type=str, default='val')
  parser.add_argument('--path_ix', type=int, default=0)
  args = parser.parse_args()

  # make dirs
  if not osp.isdir('cache/reports/'): os.makedirs('cache/reports/')
  if not osp.isdir('cache/results/imitation'): os.makedirs('cache/results/imitation')
  args.report_txt = 'cache/reports/imitation.txt'
  args.result_json = 'cache/results/imitation/%s_%s_p%s.json' % (args.id, args.split, args.path_ix)

  # run
  main(args)
