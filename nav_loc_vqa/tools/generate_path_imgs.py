"""
This gonna occupy huge amount of disk.
For each path (52 steps on average), we get RGB, depth, semantics, and cube_map for each position.
house.tid.tid.h5 will be saved for each path.
For simplicity, we only use the 1st path on default.
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
from House3D import objrender, Environment, load_config

# config
cfg = {}
house3d_metadata_dir = 'pyutils/House3D/House3D/metadata'
suncg_data_dir = 'data/SUNCGdata'
cfg['colorFile'] = osp.join(house3d_metadata_dir, 'colormap_fine.csv')
cfg['roomTargetFile'] = osp.join(house3d_metadata_dir, 'room_target_object_map.csv')
cfg['modelCategoryFile'] = osp.join(house3d_metadata_dir, 'ModelCategoryMapping.csv')
cfg['prefix'] = osp.join(suncg_data_dir, 'house')

def render(env, pos, mode='rgb'):
  """
  pos = (x, y, z, yaw) in coord
  mode could be rgb, depth, etc
  Return img of size (h, w, c).
  """
  env.cam.pos.x = pos[0]
  env.cam.pos.y = pos[1]
  env.cam.pos.z = pos[2]
  env.cam.yaw = pos[3]
  env.cam.updateDirection()
  img = env.render(mode=mode)
  return img

def render_cube_map(env, pos, mode='rgb'):
  """
  The rendered img is of size (h, w*6, 3), including 4 panorama and 2 ceiling/floor images
  We return img (4, h, w, c) 
  """
  env.cam.pos.x = pos[0]
  env.cam.pos.y = pos[1]
  env.cam.pos.z = pos[2]
  env.cam.yaw = pos[3]
  env.cam.updateDirection()
  cube_map = env.render_cube_map(mode=mode)  # (h, wx6, 3)
  w = cube_map.shape[1] // 6
  assert cube_map.shape[1] / 6 == w
  imgs = []
  for i in range(4):  # we only wanna 4 panoroma images
    imgs.append(cube_map[:, i*w:(i+1)*w, :])
  return np.array(imgs)

def get_positions(path):
  """
  Given one path = {positions, actions, best_iou, ordered}, note positions is a list of segment path ,
  we compute:
  - positions = [pos, pos, pos, ...], each is (x, y, z, yaw)
  - actions = [act, act, act, ...] whose last token is <STOP> (3)
  - key_ixs = [ix, ix], where agent at positions[ix] faces the target objects.
  - ordered = True or False
  """
  positions = [pos for seg_path in path['positions'] for pos in seg_path[1:]]
  positions.insert(0, path['positions'][0][0])  # add the spawned point
  actions = [act for seg_path in path['actions'] for act in seg_path[:-1]]
  actions.append(3) # add <STOP> in the end
  key_ixs, key_pos = [], []
  total_len = 0
  for i, seg_path in enumerate(path['positions'][:-1]):
    # last seg_path is toward end, no need being considered for key_ixs
    total_len += len(seg_path)
    key_ixs.append(total_len-i-1)
    key_pos.append(seg_path[-1])
  # check
  for ix, pos in zip(key_ixs, key_pos):
    assert positions[ix] == pos
  return positions, actions, key_ixs, path['ordered']

def cache_path_images(hid_to_qns, args):
  # make last several key frames of navigation
  api_thread = objrender.RenderAPIThread(w=args.render_width, h=args.render_height)

  # make frames
  problematic_qids = []
  for hid, qns in tqdm(hid_to_qns.items()):
    print('Processing:', hid)
    # initialize environment
    env = Environment(api_thread, hid, cfg, ColideRes=500)
    # rgb, semantic, cube_rgb (Licheng: assume pre-trained cnn is good at cnn -> depth)
    for qn in qns:
      # path json
      path_name = qn['path']+'.json'
      path_file = osp.join(args.shortest_path_dir, path_name)
      paths = json.load(open(path_file))  # list of [{positions, actions, best_iou, ordered}]
      # path h5
      output_h5 = osp.join(args.output_dir, qn['path']+'.h5')
      if osp.exists(output_h5):
        continue
      f = h5py.File(output_h5, 'w')
      # f = h5py.File(output_h5, 'r+')
      # if 'ordered0' in list(f.keys()):
      #   f.close()
      #   continue
      # render
      num_paths = min(args.num_paths, len(paths))
      for pix in range(num_paths):
        path = paths[pix]  # on default we use the first sampled path
        positions, actions, key_ixs, ordered = get_positions(path)
        ego_rgb = np.zeros((len(positions), args.render_height, args.render_width, 3))  # (n, h, w, 3)
        # ego_depth = np.zeros((len(positions), args.render_height, args.render_width, 2))  # (n, h, w, 2)
        ego_sem = np.zeros((len(positions), args.render_height, args.render_width, 3))  # (n, h, w, 3)
        cube_rgb = np.zeros((len(positions), 4, args.render_height, args.render_width, 3))  # (n, 4, h, w, 3)
        for i, pos in enumerate(positions):
          ego_rgb[i] = render(env, pos, mode='rgb')
          # ego_depth[i] = render(env, pos, mode='depth')
          ego_sem[i] = render(env, pos, mode='semantic')
          cube_rgb[i] = render_cube_map(env, pos, mode='rgb')
        # save
        f.create_dataset('ego_rgb%s'%pix, dtype=np.uint8, data=ego_rgb)
        # f.create_dataset('ego_depth%s'%pix, dtype=np.uint8, data=ego_depth)
        f.create_dataset('ego_sem%s'%pix, dtype=np.uint8, data=ego_sem)
        f.create_dataset('cube_rgb%s'%pix, dtype=np.uint8, data=cube_rgb)
        f.create_dataset('key_ixs%s'%pix, dtype=np.uint8, data=np.array(key_ixs))
        # f.create_dataset('ordered%s'%pix, dtype=np.uint8, data=np.array([ordered]))
        f.create_dataset('positions%s'%pix, dtype=np.float32, data=np.array(positions))
        f.create_dataset('actions%s'%pix, dtype=np.uint8, data=np.array(actions))
      f.create_dataset('num_paths', dtype=np.uint8, data=np.array([num_paths]))
      f.close()
      print('%s cached.' % output_h5)

def main(args):
  # load questions = [{question, answer, type, bbox, id, house, entropyj}]
  questions = json.load(open(args.input_json))
  Questions = {qn['id']: qn for qn in questions}
  hid_to_qns = {}
  for qn in questions:
    house_id = qn['house']
    if house_id not in hid_to_qns: hid_to_qns[house_id] = []
    hid_to_qns[house_id].append(qn)
  print('%s questions loaded from %s.' % (len(questions), args.input_json))
  print('%s houses are there.' % len(hid_to_qns))

  # render path images
  cache_path_images(hid_to_qns, args)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_json', default='data/question-gen-outputs/questions_mt_paths_nearby_source_best_view_program.json', help='filtered(sampled) questions json')
  parser.add_argument('--shortest_path_dir', default='data/shortest-paths-mt/nearby_source_best_view', type=str, help='directory saving sampled paths: qid.json')
  parser.add_argument('--output_dir', default='cache/path_images', help='navigation images directory')
  parser.add_argument('--num_paths', default=3, help='number of paths being sampled')
  # renderer  
  parser.add_argument('--render_width', type=int, default=224, help='rendered image width')
  parser.add_argument('--render_height', type=int, default=224, help='rendered image height')
  parser.add_argument('--seed', default=24, type=int, help='random seed')
  args = parser.parse_args()

  # run
  if not osp.isdir(args.output_dir):
    os.makedirs(args.output_dir)
  main(args)
