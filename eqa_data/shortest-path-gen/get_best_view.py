"""
This code samples the best view for each object mentioned by ``question-gen-outputs/questions_pruned_v2.json''

We sample some points around target object, which is between [2, 10] on its connMap.
Then we look for a good view, i.e., object has high iou with the center region of ego-view.
We use this found point as the ideal point for answering the question.
"""
import argparse
import random
import time
import numpy as np
import os, sys, json
import os.path as osp

from tqdm import tqdm
from pprint import pprint

from queue import Queue
from threading import Thread, Lock

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../pyutils/House3D'))
from House3D import objrender, Environment, load_config
from House3D.core import local_create_house

sys.path.insert(0, osp.join(this_dir, '../pyutils'))
from house3d import House3DUtils


def get_best_view_points(h3d, obj_id, args):
  """
  Inputs:
  - h3d    : house3d instance of some house_id
  - obj_id : object_id
  - args   : min_conn_dist, max_conn_dist, num_samples
  Return:
  - points : list of (x, 1, z, yaw), where x and z are in coords.
  - ious   : list of ious, between 0 and 1
  """
  # obj info
  obj = h3d.objects[obj_id]
  h3d.set_target_object(obj)
  obj_conn_map = h3d.env.house.connMapDict[obj_id][0]
  obj_point_cands = np.argwhere( (obj_conn_map > args.min_conn_dist) & (obj_conn_map <= args.max_conn_dist) )
  # don't search too many for saving time  
  if obj_point_cands.shape[0] > args.num_samples:
    perm = np.random.permutation(obj_point_cands.shape[0])[:args.num_samples]
    obj_point_cands = obj_point_cands[perm]
  # traverse
  points, ious, yaws = [], [], []
  for obj_point in obj_point_cands:
    obj_view, obj_iou, obj_mask = h3d._get_best_yaw_obj_from_pos(obj_id, obj_point, height=1., use_iou=True)
    x, z = h3d.env.house.to_coor(obj_point[0], obj_point[1])
    points.append([x, 1., z, obj_view])
    ious.append(obj_iou)
  # sort 
  points, ious = np.array(points), np.array(ious)
  ixs = (-ious).argsort()  # descending order
  ious = ious[ixs]
  points = points[ixs]
  # return
  return points.tolist(), ious.tolist()

def main(house_to_qns, args):

  # config
  cfg = {}
  cfg['colorFile'] = osp.join(args.house3d_metadata_dir, 'colormap_fine.csv')
  cfg['roomTargetFile'] = osp.join(args.house3d_metadata_dir, 'room_target_object_map.csv')
  cfg['modelCategoryFile'] = osp.join(args.house3d_metadata_dir, 'ModelCategoryMapping.csv')
  cfg['prefix'] = osp.join(args.suncg_data_dir, 'house')
  for d in cfg.values():
    assert osp.exists(d), d

  # data pool
  lock = Lock()
  q = Queue()
  for i, house_id in enumerate(list(house_to_qns.keys())):
    qns = house_to_qns[house_id]
    q.put((i, house_id, qns))

  # worker
  def worker():
    api_thread = objrender.RenderAPIThread(w=224, h=224)
    while True:
      i, house_id, qns = q.get()
      print('Processing house[%s] %s/%s' % (house_id, i+1, len(house_to_qns)))
      # api_thread = objrender.RenderAPIThread(w=224, h=224)
      env = Environment(api_thread, house_id, cfg, ColideRes=args.colide_resolution)
      build_graph = True
      if osp.exists(osp.join(args.graph_dir, env.house.house['id']+'.pkl')):
        build_graph = False
      h3d = House3DUtils(env, build_graph=build_graph, graph_dir=args.graph_dir, 
              target_obj_conn_map_dir=args.target_obj_conn_map_dir)

      # objects in the house of interest
      object_ids = []
      for qn in qns:
        if 'object' in qn['type']:
          for bbox in qn['bbox']:
            assert bbox['type'] == 'object'
            object_ids.append(bbox['id'])
      object_ids = list(set(object_ids))

      # sample good-view points for each object
      for obj_id in tqdm(object_ids):
        save_file = osp.join(args.output_dir, house_id+'_'+obj_id+'.json')
        if osp.exists(save_file):
          continue
        # run and save        
        points, ious = get_best_view_points(h3d, obj_id, args)
        json.dump({'points': points, 'ious': ious}, open(save_file, 'w'))
    
      q.task_done()

  for i in range(args.num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()

  return True


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--question_json', default='cache/question-gen-outputs/questions_pruned_mt_with_conn.json')
  parser.add_argument('--house3d_metadata_dir', default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--suncg_data_dir', default='data/SUNCGdata')
  parser.add_argument('--graph_dir', default='cache/3d-graphs', help='directory for saving graphs')
  parser.add_argument('--target_obj_conn_map_dir', default='cache/target-obj-conn-maps')
  parser.add_argument('--colide_resolution', default=500, type=int, help='house grid resolution')
  parser.add_argument('--seed', default=24, type=int, help='random seed')
  parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
  # sample argument
  parser.add_argument('--num_samples', default=50, type=int, help='number of sampled points near target object')
  parser.add_argument('--max_conn_dist', default=10, type=int, help='max #steps going to target object')
  parser.add_argument('--min_conn_dist', default=2, type=int, help='min #steps going to target object')
  # output argument
  parser.add_argument('--output_dir', default='cache/target-obj-bestview-pos', type=str, help='output directory')
  args = parser.parse_args()

  # set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)

  # output directory
  if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)
  
  # load questions
  questions = json.load(open(args.question_json, 'r'))
  house_to_qns = {}
  for qn in questions:
    house_id = qn['house']
    if house_id not in house_to_qns: house_to_qns[house_id] = []
    house_to_qns[house_id].append(qn)

  # run
  done = main(house_to_qns, args)
  print(done)

