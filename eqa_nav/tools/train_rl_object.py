import h5py
import time
import argparse
import numpy as np
import os, sys, json
import os.path as osp

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import _init_paths
from nav.loaders.nav_reinforce_loader import NavReinforceDataset
from nav.models.navigator import Navigator

from nav.reinforce.eval_process import eval
from nav.reinforce.train_process import train
from nav.reinforce.imitation_process import imitation

def main(args):

  mp.set_start_method('forkserver', force=True)
  args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
  args.gpus = [int(x) for x in args.gpus]

  # set up shared_model
  checkpoint_path = osp.join(args.checkpoint_dir, '%s.pth' % args.start_from)
  checkpoint = torch.load(checkpoint_path)
  shared_nav_model = Navigator(checkpoint['opt'])
  shared_nav_model.load_state_dict(checkpoint['model_state'])
  shared_nav_model.cpu()
  shared_nav_model.share_memory()
  print('shared_nav_model set up.')
  # some arguments need to be copied from start_from
  args.use_action = checkpoint['opt']['use_action']
  args.nav_types = ['object']

  # processes
  processes = []
  counter = mp.Value('i', 0)
  lock = mp.Lock()

  # train
  for rank in range(args.num_processes):
    p = mp.Process(target=train, args=(rank, args, shared_nav_model, counter, lock))
    p.start()
    processes.append(p)

  # imitation
  p = mp.Process(target=imitation, args=(args.num_processes, args, shared_nav_model, counter))
  p.start()
  processes.append(p)

  # eval
  p = mp.Process(target=eval, args=(args.num_processes+1, args, shared_nav_model, counter, 'val'))
  p.start()
  processes.append(p)

  # join
  for p in processes:
    p.join()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  # Data input settings
  parser.add_argument('--data_json', type=str, default='cache/prepro/reinforce/data.json')
  parser.add_argument('--data_h5', type=str, default='cache/prepro/reinforce/data.h5')
  parser.add_argument('--imitation_data_json', type=str, default='cache/prepro/imitation/data.json')
  parser.add_argument('--imitation_data_h5', type=str, default='cache/prepro/imitation/data.h5')
  parser.add_argument('--path_feats_dir', type=str, default='cache/path_feats')
  parser.add_argument('--path_images_dir', type=str, default='cache/path_images')
  parser.add_argument('--target_obj_conn_map_dir', type=str, default='data/target-obj-conn-maps')
  parser.add_argument('--pretrained_cnn_path', type=str, default='cache/hybrid_cnn.pt')
  parser.add_argument('--house_meta_dir', type=str, default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--house_data_dir', type=str, default='data/SUNCGdata/house')
  parser.add_argument('--checkpoint_dir', type=str, default='output/nav_object')
  parser.add_argument('--seed', type=int, default=24)
  # multiprocess settings
  parser.add_argument('--num_processes', type=int, default=12)
  # log settings
  parser.add_argument('--max_epochs', type=int, default=500)
  parser.add_argument('--num_iters_per_epoch', type=int, default=1000)
  parser.add_argument('--tb_dir', type=str, default='log_dir/nav_object')
  parser.add_argument('--log_dir', type=str, default='log_dir/nav_object')
  # Navigator settings
  parser.add_argument('--shortest_path_ratio', type=float, default=1.0)
  parser.add_argument('--max_episode_length', type=int, default=80)
  parser.add_argument('--max_threads_per_gpu', type=int, default=1)
  parser.add_argument('--mult_increasing_per_iters', type=int, default=2500)
  parser.add_argument('--max_seq_length', type=int, default=50, help='max_seq_length')
  parser.add_argument('--rnn_type', type=str, default='lstm')
  parser.add_argument('--rnn_size', type=int, default=256)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--rnn_dropout', type=float, default=0.1)
  parser.add_argument('--fc_dropout', type=float, default=0.0)
  parser.add_argument('--seq_dropout', type=float, default=0.0)
  parser.add_argument('--fc_dim', type=int, default=64)
  parser.add_argument('--act_dim', type=int, default=64)
  # Output settings
  parser.add_argument('--start_from', type=str, default='im0')
  parser.add_argument('--id', type=str, default='rl0')
  # Optimizer
  parser.add_argument('--batch_size', type=int, default=20, help='batch size in number of questions per batch')
  parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
  parser.add_argument('--lr_decay', type=int, default=1, help='if decay learning rate')
  parser.add_argument('--learning_rate_decay_start', type=int, default=8000, help='at what iters to start decaying learning rate')
  parser.add_argument('--learning_rate_decay_every', type=int, default=8000, help='every how many iters thereafter to drop LR by half')
  parser.add_argument('--im_learning_rate_decay_start', type=int, default=8000, help='learning rate decay start on Imitation')
  parser.add_argument('--im_learning_rate_decay_every', type=int, default=8000, help='learning rate decay every on Imitation')
  parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
  parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
  parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
  parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for l2 regularization')
  args = parser.parse_args()

  # update log_dir and tb_dir
  args.log_dir = osp.join(args.log_dir, args.id)
  args.tb_dir = osp.join(args.tb_dir, args.id)
  if not osp.exists(args.log_dir): os.makedirs(args.log_dir)
  if not osp.exists(args.tb_dir): os.makedirs(args.tb_dir)

  # main
  main(args)
