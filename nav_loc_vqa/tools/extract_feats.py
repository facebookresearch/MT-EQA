"""
We have run ``tools/generate_path_imgs.py`` for path generation.
Let's now extract the features.
For each h5 file, we only extracted features for rgb-related images.
"""
import h5py
import os
import os.path as osp
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import _init_paths
from vqa.models.cnn import MultitaskCNN

import torch


def main(args):
  
  # load questions = [{question, answer, type, bbox, id, house, entropy}]
  questions = json.load(open(args.input_json))
  # h5 files
  h5_files = [f for f in os.listdir(args.path_images_dir) if '.h5' in f]
  print('We found %s path_images h5_files for %s quesitons.' % (len(h5_files), len(questions)))
  print('Normal if we have fewer paths than questions, as some of the paths are shared.')

  # set up CNN
  cnn_kwargs = {'num_classes': 191, 'pretrained': True, 'checkpoint_path': args.pretrained_cnn_path}
  cnn = MultitaskCNN(**cnn_kwargs)
  cnn.cuda()
  cnn.eval()
  print('cnn set up.')

  # forward h5's rgb images through CNN
  # h5_files = h5_files[:8000]
  h5_files = h5_files[8000:]
  batch_size = args.batch_size
  for h5_file in tqdm(h5_files):
    # output h5
    if osp.exists(osp.join(args.output_dir, h5_file)):
      print('%s exists.' % osp.join(args.output_dir, h5_file))
      continue
    f = h5py.File(osp.join(args.output_dir, h5_file), 'w')

    # forward
    data = h5py.File(osp.join(args.path_images_dir, h5_file), 'r')
    num_paths = data['num_paths'][0]
    for i in range(int(num_paths)):
      ego_rgbs = data['ego_rgb%s'%i][...]  # (n, 224, 224, 3)
      ego_rgbs = ego_rgbs.astype(np.float32) / 255.
      ego_rgbs = ego_rgbs.transpose(0,3,1,2)
      ego_rgbs = torch.from_numpy(ego_rgbs)
      num_frames = ego_rgbs.shape[0]
      ego_feats = np.zeros((num_frames, 32, 10, 10), np.float32)  
      for b in range(0, num_frames, batch_size):
        _, _, batch_feats  = cnn.extract_feats(ego_rgbs[b:b+batch_size].cuda())
        ego_feats[b:b+batch_size] = batch_feats.data.cpu().numpy()
      # save
      f.create_dataset('ego_rgb%s'%i, dtype=np.float32, data=ego_feats)
    
    for i in range(int(num_paths)):
      cube_rgbs = data['cube_rgb%s'%i][...]  # (n, 4, 224, 224, 3)
      cube_rgbs = cube_rgbs.astype(np.float32) / 255.
      cube_rgbs = cube_rgbs.transpose(0, 1, 4, 2, 3)  # (n, 4, 3, 224, 224)
      cube_rgbs = torch.from_numpy(cube_rgbs)
      num_frames = cube_rgbs.shape[0]
      cube_feats = np.zeros((num_frames, 4, 32, 10, 10), np.float32)
      for b in range(0, num_frames, batch_size):
        batch_cube_rgbs = cube_rgbs[b:b+batch_size].reshape(-1, 3, 224, 224)  # (batch x 4, 3, 224, 224)
        _, _, batch_cube_feats = cnn.extract_feats(batch_cube_rgbs.cuda())
        batch_cube_feats = batch_cube_feats.view(-1, 4, 32, 10, 10)  # (batch, 4, 32, 10, 10)
        cube_feats[b:b+batch_size] = batch_cube_feats.data.cpu().numpy()
      # save
      f.create_dataset('cube_rgb%s'%i, dtype=np.float32, data=cube_feats)

    data.close()
    f.close()
    print('%s cached.' % osp.join(args.output_dir, h5_file))

  print('Done.')


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--pretrained_cnn_path', type=str, default='cache/hybrid_cnn.pt')
  parser.add_argument('--input_json', default='data/question-gen-outputs/questions_mt_paths_nearby_source_best_view.json', help='filtered(sampled) questions json')
  parser.add_argument('--path_images_dir', default='cache/path_images', type=str, help='directory saving sampled paths: qid.json')
  parser.add_argument('--batch_size', default=30, type=int, help='batch size')
  parser.add_argument('--output_dir', default='cache/path_feats', help='navigation images directory')
  # renderer  
  args = parser.parse_args()

  # run
  if not osp.isdir(args.output_dir):
    os.makedirs(args.output_dir)
  main(args)
