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
  return predictions, overall_nll, overall_teacher_forcing_acc, overall_mse, Nav_nll, Nav_teacher_forcing_acc
  
def main(args):
  # make output directory
  if args.checkpoint_dir is None: 
    args.checkpoint_dir = 'output/nav_room'
  if not osp.isdir(args.checkpoint_dir): 
    os.makedirs(args.checkpoint_dir)

  # set random seed
  random.seed(args.seed)
  np.random.randn(args.seed)
  torch.manual_seed(args.seed)

  # set up loaders
  train_loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'train',
    'max_seq_length': args.max_seq_length,
    'requires_imgs': False,
    'nav_types': ['room'],
    'question_types': ['all'],
  }
  train_dataset = NavImitationDataset(**train_loader_kwargs)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
  val_loader_kwargs = {
    'data_json': args.data_json,
    'data_h5': args.data_h5,
    'path_feats_dir': args.path_feats_dir,
    'path_images_dir': args.path_images_dir,
    'split': 'val',
    'max_seq_length': args.max_seq_length,
    'requires_imgs': False,
    'nav_types': ['room'],
    'question_types': ['all'],
  }
  val_dataset = NavImitationDataset(**val_loader_kwargs)

  # set up models
  opt = vars(args)
  opt['act_to_ix'] = train_dataset.act_to_ix
  opt['num_actions'] = len(opt['act_to_ix'])
  model = Navigator(opt)
  model.cuda()
  print('navigator set up.')

  # set up criterions
  nll_crit = SeqModelCriterion().cuda()
  mse_crit = MaskedMSELoss().cuda()

  # set up optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                betas=(args.optim_alpha, args.optim_beta), eps=args.optim_epsilon,
                weight_decay=args.weight_decay)
  
  # resume from checkpoint
  infos = {}
  iters = infos.get('iters', 0)
  epoch = infos.get('epoch', 0)
  val_nll_history = infos.get('val_nll_history', {})
  val_mse_history = infos.get('val_mse_history', {})
  val_teacher_forcing_acc_history = infos.get('val_teacher_forcing_acc_history', {})
  val_nav_object_nll_history = infos.get('val_nav_object_nll_history', {})
  val_nav_object_teacher_forcing_acc_history = infos.get('val_nav_object_teacher_forcing_acc_history', {})
  val_nav_room_nll_history = infos.get('val_nav_room_nll_history', {})
  val_nav_room_teacher_forcing_acc_history = infos.get('val_nav_room_teacher_forcing_acc_history', {})
  loss_history = infos.get('loss_history', {})
  nll_loss_history = infos.get('nll_loss_history', {})
  mse_loss_history = infos.get('mse_loss_history', {})
  lr = infos.get('lr', args.learning_rate)
  best_val_score, best_val_acc, best_predictions = None, None, None

  # start training
  while iters <= args.max_iters:
    print('Starting epoch %d' % epoch)
    # reset seq_length
    if args.use_curriculum:
      # assume we need 4 epochs to get full seq_length
      seq_length = min((args.max_seq_length // 4) ** (epoch+1), args.max_seq_length)
      train_dataset.reset_seq_length(seq_length)
    else:
      seq_length = args.max_seq_length
    # train
    for batch in train_loader:
      # set mode
      model.train()
      # zero gradient
      optimizer.zero_grad()
      # batch = {qid, path_ix, house, id, type, phrase, phrase_emb, ego_feats, next_feats, res_feats,
      #  action_inputs, action_outputs, action_masks, ego_imgs}
      ego_feats = batch['ego_feats'].cuda()  # (n, L, 3200)
      phrase_embs = batch['phrase_emb'].cuda()  # (n, 300)
      action_inputs = batch['action_inputs'].cuda()   # (n, L)
      action_outputs = batch['action_outputs'].cuda() # (n, L)
      action_masks = batch['action_masks'].cuda()  # (n, L)
      # forward
      # - logprobs (n, L, #actions)
      # - output_feats (n, L, rnn_size)
      # - pred_feats (n, L, 3200) or None
      logprobs, _, pred_feats, _ = model(ego_feats, phrase_embs, action_inputs)  
      nll_loss = nll_crit(logprobs, action_outputs, action_masks)
      mse_loss = 0
      if args.use_next:
        next_feats = batch['next_feats'].cuda()  # (n, L, 3200)
        mse_loss = mse_crit(pred_feats, next_feats, action_masks)
      if args.use_residual:
        res_feats = batch['res_feats'].cuda()  # (n, L, 3200)
        mse_loss = mse_crit(pred_feats, res_feats, action_masks)
      loss = nll_loss + args.mse_weight * mse_loss
      # backward
      loss.backward()
      model_utils.clip_gradient(optimizer, args.grad_clip)
      optimizer.step()

      # training log
      if iters % args.losses_log_every == 0:
        loss_history[iters] = loss.item()
        nll_loss_history[iters] = nll_loss.item()
        mse_loss_history[iters] = mse_loss.item() if (args.use_next or args.use_residual) else 0
        print('iters[%s]epoch[%s], train_loss=%.3f (nll_loss=%.3f, mse_loss=%.3f) lr=%.2E, cur_seq_length=%s' % \
          (iters, epoch, loss_history[iters], nll_loss_history[iters], mse_loss_history[iters], lr, train_loader.dataset.cur_seq_length))

      # decay learning rate
      if args.learning_rate_decay_start > 0 and iters > args.learning_rate_decay_start:
        frac = (iters - args.learning_rate_decay_start) / args.learning_rate_decay_every
        decay_factor = 0.1 ** frac
        lr = args.learning_rate * decay_factor
        model_utils.set_lr(optimizer, lr)

      # evaluate
      if iters % args.save_checkpoint_every == 0:
        print('Checking validation ...')
        predictions, overall_nll, overall_teacher_forcing_acc, overall_mse, Nav_nll, Nav_teacher_forcing_acc = \
          evaluate(val_dataset, model, nll_crit, mse_crit, opt)
        val_nll_history[iters] = overall_nll
        val_teacher_forcing_acc_history[iters] = overall_teacher_forcing_acc
        val_mse_history[iters] = overall_mse
        val_nav_object_nll_history[iters] = Nav_nll['object']
        val_nav_object_teacher_forcing_acc_history[iters] = Nav_teacher_forcing_acc['object']
        val_nav_room_nll_history[iters] = Nav_nll['room']
        val_nav_room_teacher_forcing_acc_history[iters] = Nav_teacher_forcing_acc['room']

        # save model if best
        # consider all three accuracy, perhaps a better weighting is needed.
        current_score = -overall_nll
        if best_val_score is None or current_score > best_val_score:
          best_val_score = current_score
          best_val_acc = overall_teacher_forcing_acc
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
        infos['nll_loss_history'] = nll_loss_history
        infos['mse_loss_history'] = mse_loss_history
        infos['val_nll_history'] = val_nll_history
        infos['val_teacher_forcing_acc_history'] = val_teacher_forcing_acc_history
        infos['val_mse_history'] = val_mse_history
        infos['val_nav_object_nll_history'] = val_nav_object_nll_history
        infos['val_nav_object_teacher_forcing_acc_history'] = val_nav_object_teacher_forcing_acc_history
        infos['val_nav_room_nll_history'] = val_nav_room_nll_history
        infos['val_nav_room_teacher_forcing_acc_history'] = val_nav_room_teacher_forcing_acc_history
        infos['best_val_score'] = best_val_score
        infos['best_val_acc'] = best_val_acc
        infos['best_predictions'] = predictions if best_predictions is None else best_predictions
        infos['opt'] = vars(args)
        infos['act_to_ix'] = train_dataset.act_to_ix
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
  parser.add_argument('--data_h5', type=str, default='cache/prepro/data.h5')
  parser.add_argument('--path_feats_dir', type=str, default='cache/path_feats')
  parser.add_argument('--path_images_dir', type=str, default='cache/path_images')
  parser.add_argument('--checkpoint_dir', type=str, default='output/nav_room')
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--seed', type=int, default=24)
  parser.add_argument('--start_from', type=str, default=None)
  # Navigator settings
  parser.add_argument('--max_seq_length', type=int, default=100, help='max_seq_length')
  parser.add_argument('--rnn_type', type=str, default='lstm')
  parser.add_argument('--rnn_size', type=int, default=256)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--rnn_dropout', type=float, default=0.1)
  parser.add_argument('--fc_dropout', type=float, default=0.0)
  parser.add_argument('--seq_dropout', type=float, default=0.0)
  parser.add_argument('--fc_dim', type=int, default=64)
  parser.add_argument('--act_dim', type=int, default=64)
  parser.add_argument('--use_action', dest='use_action', action='store_true', help='if input previous action')
  parser.add_argument('--use_residual', dest='use_residual', action='store_true', help='if predict the residual featuer')
  parser.add_argument('--use_next', dest='use_next', action='store_true', help='if predict next image feature')
  parser.add_argument('--use_curriculum', dest='use_curriculum', action='store_true', help='if use curriculum')
  # Output settings
  parser.add_argument('--id', type=str, default='im0')
  parser.add_argument('--save_checkpoint_every', type=str, default=2000, help='how often to save a model checkpoint')
  parser.add_argument('--losses_log_every', type=int, default=25)
  # Optimizer
  parser.add_argument('--mse_weight', type=float, default=1.0)
  parser.add_argument('--max_iters', type=int, default=20000, help='max number of iterations to run')
  parser.add_argument('--batch_size', type=int, default=40, help='batch size in number of questions per batch')
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
