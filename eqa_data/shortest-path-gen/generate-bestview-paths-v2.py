"""
This code generate shortest-paths for each question.
Paths start from and end at best-view points.

Sampled paths which are list of [{positions, actions, best_ious, ordered}] are saved
as "cache/shortest-paths-v2/qid.json".

{'positions': positions, 'actions': actions, 'best_iou': {obj1_name: obj1_iou, obj2_name: obj2_iou}, 'ordered': ordered}
"""
import argparse
import random
import time
import numpy as np
import os, sys, json
import os.path as osp
from tqdm import tqdm

from queue import Queue
from threading import Thread, Lock

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../pyutils/House3D'))
from House3D import objrender, Environment, load_config
from House3D.core import local_create_house

sys.path.insert(0, osp.join(this_dir, '../pyutils'))
from house3d import House3DUtils


def get_5points_for_3objects(h3d, obj1_id, obj2_id, obj3_id, args):
  """
  Return 5 grid points (source_point, obj1_point, obj2_point, obj3_point, end_point) for 3 objects
  mentioned by the question, e.g., ``is <obj1> closer to <obj2> than <obj3>?''
  """
  # three objects
  house_id = h3d.env.house.house['id']
  obj1, obj2, obj3 = h3d.objects[obj1_id], h3d.objects[obj2_id], h3d.objects[obj3_id]

  h3d.set_target_object(obj1)
  obj1_conn_map = h3d.env.house.connMapDict[obj1_id][0]
  obj1_best_view_info = json.load(open(osp.join(args.target_obj_best_view_dir, house_id+'_'+obj1['id']+'.json')))  # {points, ious}
  obj1_points, obj1_ious = obj1_best_view_info['points'], obj1_best_view_info['ious']

  h3d.set_target_object(obj2)
  obj2_conn_map = h3d.env.house.connMapDict[obj2_id][0]
  obj2_best_view_info = json.load(open(osp.join(args.target_obj_best_view_dir, house_id+'_'+obj2['id']+'.json')))  # {points, ious}
  obj2_points, obj2_ious = obj2_best_view_info['points'], obj2_best_view_info['ious']

  h3d.set_target_object(obj3)
  obj3_conn_map = h3d.env.house.connMapDict[obj3_id][0]
  obj3_best_view_info = json.load(open(osp.join(args.target_obj_best_view_dir, house_id+'_'+obj3['id']+'.json')))  # {points, ious}
  obj3_points, obj3_ious = obj3_best_view_info['points'], obj3_best_view_info['ious']
  
  # source point
  if args.source_mode == 'nearby':
    # source point (p \in [target_dist, source_dist] of obj1)
    source_point_cands = np.argwhere((obj1_conn_map > args.min_dist_thresh) & (obj1_conn_map <= args.source_dist_thresh) )
  elif args.source_mode == 'random':
    source_point_cands = np.argwhere((obj1_conn_map > 0) & (obj2_conn_map > 0) & (obj3_conn_map > 0))
  else:
    raise NotImplementedError
  source_point_idx = np.random.choice(source_point_cands.shape[0])
  source_point = (source_point_cands[source_point_idx][0], source_point_cands[source_point_idx][1], np.random.choice(h3d.angles))

  # object 1 point
  obj1_point_cands = [(obj1_points[ix], iou) for ix, iou in enumerate(obj1_ious) if iou > args.good_iou]
  if len(obj1_point_cands) == 0:
    obj1_point_cands = [(obj1_points[0], obj1_ious[0])] if obj1_ious[0] > args.min_iou else []
  assert len(obj1_point_cands) > 0, 'best iou is below %.2f' % args.min_iou
  obj1_point, obj1_iou = obj1_point_cands[np.random.choice(len(obj1_point_cands))]  # (x, y, z, yaw) in coord
  x, z = h3d.env.house.to_grid(obj1_point[0], obj1_point[2])
  obj1_point = (x, z, obj1_point[-1])

  # object 2 point
  obj2_point_cands = [(obj2_points[ix], iou) for ix, iou in enumerate(obj2_ious) if iou > args.good_iou]
  if len(obj2_point_cands) == 0:
    obj2_point_cands = [(obj2_points[0], obj2_ious[0])] if obj2_ious[0] > args.min_iou else []
  assert len(obj2_point_cands) > 0, 'best iou is below %.2f' % args.min_iou
  obj2_point, obj2_iou = obj2_point_cands[np.random.choice(len(obj2_point_cands))]  # (x, y, z, yaw) in coord
  x, z = h3d.env.house.to_grid(obj2_point[0], obj2_point[2])
  obj2_point = (x, z, obj2_point[-1])

  # object 3 point
  obj3_point_cands = [(obj3_points[ix], iou) for ix, iou in enumerate(obj3_ious) if iou > args.good_iou]
  if len(obj3_point_cands) == 0:
    obj3_point_cands = [(obj3_points[0], obj3_ious[0])] if obj3_ious[0] > args.min_iou else []
  assert len(obj3_point_cands) > 0, 'best iou is below %.2f' % args.min_iou
  obj3_point, obj3_iou = obj3_point_cands[np.random.choice(len(obj3_point_cands))]  # (x, y, z, yaw) in coord
  x, z = h3d.env.house.to_grid(obj3_point[0], obj3_point[2])
  obj3_point = (x, z, obj3_point[-1])

  # end point
  if args.source_mode == 'nearby':
    # end point (p \in [target_dist, source_dist] of obj3)
    end_point_cands = np.argwhere((obj3_conn_map > args.min_dist_thresh) & (obj3_conn_map <= args.end_dist_thresh) )
  elif args.source_mode == 'random':
    end_point_cands = np.argwhere((obj1_conn_map > 0) & (obj2_conn_map > 0) & (obj3_conn_map > 0))
  else:
    raise NotImplementedError
  end_point_idx = np.random.choice(end_point_cands.shape[0])
  end_point = (end_point_cands[end_point_idx][0], end_point_cands[end_point_idx][1], np.random.choice(h3d.angles))

  return source_point, obj1_point, obj2_point, obj3_point, end_point, obj1_iou, obj2_iou, obj3_iou

def get_4points_for_2objects(h3d, obj1_id, obj2_id, args):
  """
  Return 4 grid points (source_point, obj1_point, obj2_point, end_point) for two objects
  mentioned by question, e.g., ``is color of <obj1> same as <obj2>?''
  """
  # two objects
  house_id = h3d.env.house.house['id']
  obj1, obj2 = h3d.objects[obj1_id], h3d.objects[obj2_id]

  h3d.set_target_object(obj1)
  obj1_conn_map = h3d.env.house.connMapDict[obj1_id][0]
  obj1_best_view_info = json.load(open(osp.join(args.target_obj_best_view_dir, house_id+'_'+obj1['id']+'.json')))  # {points, ious} 
  obj1_points, obj1_ious = obj1_best_view_info['points'], obj1_best_view_info['ious']

  h3d.set_target_object(obj2)
  obj2_conn_map = h3d.env.house.connMapDict[obj2_id][0]
  obj2_best_view_info = json.load(open(osp.join(args.target_obj_best_view_dir, house_id+'_'+obj2['id']+'.json')))  # {points, ious}
  obj2_points, obj2_ious = obj2_best_view_info['points'], obj2_best_view_info['ious']

  # source point
  if args.source_mode == 'nearby':
    # source target (p \in [target_dist, source_dist] of obj1)
    source_point_cands = np.argwhere((obj1_conn_map > args.min_dist_thresh) & (obj1_conn_map <= args.source_dist_thresh) )
  elif args.source_mode == 'random':
    source_point_cands = np.argwhere((obj1_conn_map > 0 ) & (obj2_conn_map > 0) )  # randomly spawned
  else:
    raise NotImplementedError
  source_point_idx = np.random.choice(source_point_cands.shape[0])
  source_point = (source_point_cands[source_point_idx][0], source_point_cands[source_point_idx][1], np.random.choice(h3d.angles))

  # object 1 point
  obj1_point_cands = [(obj1_points[ix], iou) for ix, iou in enumerate(obj1_ious) if iou > args.good_iou]
  if len(obj1_point_cands) == 0:
    obj1_point_cands = [(obj1_points[0], obj1_ious[0])] if obj1_ious[0] > args.min_iou else []
  assert len(obj1_point_cands) > 0, 'best iou is below %.2f' % args.min_iou
  obj1_point, obj1_iou = obj1_point_cands[np.random.choice(len(obj1_point_cands))]  # (x, y, z, yaw) in coord
  x, z = h3d.env.house.to_grid(obj1_point[0], obj1_point[2])
  obj1_point = (x, z, obj1_point[-1])

  # object 2 point
  obj2_point_cands = [(obj2_points[ix], iou) for ix, iou in enumerate(obj2_ious) if iou > args.good_iou]
  if len(obj2_point_cands) == 0:
    obj2_point_cands = [(obj2_points[0], obj2_ious[0])] if obj2_ious[0] > args.min_iou else []
  assert len(obj2_point_cands) > 0, 'best iou is below %.2f' % args.min_iou
  obj2_point, obj2_iou = obj2_point_cands[np.random.choice(len(obj2_point_cands))]  # (x, y, z, yaw) in coord
  x, z = h3d.env.house.to_grid(obj2_point[0], obj2_point[2])
  obj2_point = (x, z, obj2_point[-1])

  # end point
  if args.source_mode == 'nearby':
    # end target (p \in [target_dist, source_dist] of obj2)
    end_point_cands = np.argwhere((obj2_conn_map > args.min_dist_thresh) & (obj2_conn_map <= args.end_dist_thresh) )
  elif args.source_mode == 'random':
    end_point_cands = np.argwhere((obj1_conn_map > 0 ) & (obj2_conn_map > 0) )  # randomly spawned
  else:
    raise NotImplementedError
  end_point_idx = np.random.choice(end_point_cands.shape[0])
  end_point = (end_point_cands[end_point_idx][0], end_point_cands[end_point_idx][1], np.random.choice(h3d.angles))

  return source_point, obj1_point, obj2_point, end_point, obj1_iou, obj2_iou

def get_4points_for_2rooms(h3d, room1_id, room2_id, args):
  """
  Return 4 grid points (source_point, room1_point, room2_point, end_point) for two rooms
  mentioned by question, e.g., ``is room1 bigger than room2?''
  """
  # two rooms
  room1, room2 = h3d.rooms[room1_id], h3d.rooms[room2_id]
  h3d.set_target_room(room1)
  h3d.set_target_room(room2)
  room1_conn_map = h3d.env.house.connMapDict[room1_id][0]
  room2_conn_map = h3d.env.house.connMapDict[room2_id][0]

  if args.source_mode == 'nearby':
    # source target (p \in [target_dist, source_dist] of room1)
    source_point_cands = np.argwhere((room1_conn_map > args.min_dist_thresh) & (room1_conn_map <= args.source_dist_thresh) )
  elif args.source_mode == 'random':
    source_point_cands = np.argwhere((room1_conn_map > 0) & (room2_conn_map > 0) )  # randomly spawned
  else:
    raise NotImplementedError
  source_point_idx = np.random.choice(source_point_cands.shape[0])
  source_point = (source_point_cands[source_point_idx][0], source_point_cands[source_point_idx][1], np.random.choice(h3d.angles))

  # room 1 point (p \in [0, target_dist])
  room1_point_cands = np.argwhere((room1_conn_map >= 0) & (room1_conn_map <= args.min_dist_thresh) )
  room1_point_idx = np.random.choice(room1_point_cands.shape[0])
  room1_point = (room1_point_cands[room1_point_idx][0], room1_point_cands[room1_point_idx][1], np.random.choice(h3d.angles))

  # room 2 point (p \in [0, target_dist])
  room2_point_cands = np.argwhere((room2_conn_map >= 0) & (room2_conn_map <= args.min_dist_thresh) )
  room2_point_idx = np.random.choice(room2_point_cands.shape[0])
  room2_point = (room2_point_cands[room2_point_idx][0], room2_point_cands[room2_point_idx][1], np.random.choice(h3d.angles))

  if args.source_mode == 'nearby':
    # end target (p \in [target_dist, source_dist] for room2)
    end_point_cands = np.argwhere((room2_conn_map > args.min_dist_thresh) & (room2_conn_map <= args.end_dist_thresh) )
  elif args.source_mode == 'random':
    end_point_cands = np.argwhere((room1_conn_map > 0) & (room2_conn_map > 0) )  # randomly spawned
  else:
    raise NotImplementedError
  end_point_idx = np.random.choice(end_point_cands.shape[0])
  end_point = (end_point_cands[end_point_idx][0], end_point_cands[end_point_idx][1], np.random.choice(h3d.angles))

  return source_point, room1_point, room2_point, end_point


def path_to_actions(h3d, shortest_path):
  """
  return list of positions and actions, the last of which is always 'STOP'
  """
  act_q, pos_q = [], []
  for i in shortest_path:
    gx, gy = h3d.env.house.to_coor(i[0], i[1])
    yaw = i[2]
    pos_q.append([float(gx), 1.0, float(gy), float(yaw)])

  # action_map = {0: 'FRWD', 1: 'LEFT', 2: 'RGHT', 3: 'STOP'}
  for i in range(len(pos_q)-1):
    if pos_q[i][0] == pos_q[i+1][0] and pos_q[i][2] == pos_q[i+1][2]:
      # turn
      if pos_q[i][3] > 0 and pos_q[i+1][3] == -180:   # 150 -> -180
        act_q.append(2)
      elif pos_q[i][3] == -180 and pos_q[i+1][3] > 0: # -180 -> 150
        act_q.append(1)
      elif pos_q[i+1][3] > pos_q[i][3]:
        act_q.append(2)
      elif pos_q[i+1][3] < pos_q[i][3]:
        act_q.append(1)
      else:
        raise NotImplementedError
    else:
      act_q.append(0)
  act_q.append(3)
  return act_q, pos_q

def sample_paths(h3d, path_points):
  """
  Inputs:
  - path_points: list of grid points
  Return:
  list of sampled path information:
  - positions: list of list of [[(pos_x, 1.0, pos_z, yaw)]], in coordinates (not grids).
  - actions: list of list of [[actions]]
  we only try two version: source_point -> points, source_point -> points[::-1]
  most of the cases are fine, unless we have 3 points we will miss some combinations
  """
  # source --> points
  positions, actions = [], []
  for pt0, pt1 in zip(path_points[:-1], path_points[1:]):
    spath = h3d.compute_shortest_path(pt0, pt1)
    act_q, pos_q = path_to_actions(h3d, spath)
    positions.append(pos_q)
    actions.append(act_q)
  return positions, actions

def main(house_to_qns, args):

  invalid = []

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
      env = Environment(api_thread, house_id, cfg, ColideRes=args.colide_resolution)
      h3d = House3DUtils(env, build_graph=False, graph_dir=args.graph_dir,
              target_obj_conn_map_dir=args.target_obj_conn_map_dir)

      # compute shortest path for each qn
      for qn in qns:

        tic = time.time()
        samples = []

        for _ in range(args.num_samples):

          sample = {'shortest_paths': [], 'positions': []}

          if qn['type'] in ['object_color_compare_inroom', 'object_color_compare_xroom', 'object_size_compare_inroom', 'object_size_compare_xroom']:
            ntnp = '2obj4p'
            # 2 objects
            assert len(qn['bbox']) == 2
            obj1_id, obj2_id = qn['bbox'][0]['id'], qn['bbox'][1]['id']
            obj1_name, obj2_name = qn['bbox'][0]['name'], qn['bbox'][1]['name']
            # 4 points
            try:
              source_point, obj1_point, obj2_point, end_point, obj1_iou, obj2_iou = get_4points_for_2objects(h3d, obj1_id, obj2_id, args)
            except:
              print('4 points for 2 objects not found.')
              continue
            # compute shortest paths
            try:
              ordered = True
              positions, actions = sample_paths(h3d, [source_point, obj1_point, obj2_point, end_point])
              samples.append({'positions': positions, 'actions': actions, 'best_iou': {obj1_name: obj1_iou, obj2_name: obj2_iou}, 'ordered': ordered})
            except:
              print('shortest path not found for question[%s](%s).' % (qn['id'], qn['type']))
              continue

          elif qn['type'] == 'object_dist_compare_inroom':
            ntnp = '3obj5p'
            # 3 objects
            assert len(qn['bbox']) == 3
            obj1_id, obj2_id, obj3_id = qn['bbox'][0]['id'], qn['bbox'][1]['id'], qn['bbox'][2]['id']
            obj1_name, obj2_name, obj3_name = qn['bbox'][0]['name'], qn['bbox'][1]['name'], qn['bbox'][2]['name']
            # 5 points
            try:
              source_point, obj1_point, obj2_point, obj3_point, end_point, obj1_iou, obj2_iou, obj3_iou = \
                get_5points_for_3objects(h3d, obj1_id, obj2_id, obj3_id, args)
            except:
              print('5 points for 3 objects not found.')
              continue
            # compute shortest paths
            try:
              ordered = True
              positions, actions = sample_paths(h3d, [source_point, obj1_point, obj2_point, obj3_point, end_point])
              samples.append({'positions': positions, 'actions': actions, 'best_iou': {obj1_name: obj1_iou, obj2_name: obj2_iou, obj3_name: obj3_iou}, 'ordered': ordered})
            except:
              print('shortest path not found for question[%s](%s).' % (qn['id'], qn['type']))
              continue

          elif qn['type'] == 'room_size_compare':
            ntnp = '2rm4p'
            # 2 rooms
            assert len(qn['bbox']) == 2
            room1_id, room2_id = qn['bbox'][0]['id'], qn['bbox'][1]['id']
            # 4 points
            try:
              source_point, room1_point, room2_point, end_point = get_4points_for_2rooms(h3d, room1_id, room2_id, args)
            except:
              print('4 points for 2 rooms not found.')
              continue
            # compute shortest paths
            try:
              ordered = True
              positions, actions = sample_paths(h3d, [source_point, room1_point, room2_point, end_point])
              samples.append({'positions': positions, 'actions': actions, 'ordered': ordered})
            except:
              print('shortest path not found for question[%s][%s].' % (qn['id'], qn['type']))
              continue

        # save
        if len(samples) == 0:
          invalid.append(qn['id'])  # do not use += here!
        else:
          print('%s [%s] samples for question[%s] in %.2fs.' % (len(samples), ntnp, qn['id'], time.time()-tic))
          fname = [str(qn['house'])] + [str(box['id'])for box in qn['bbox']]
          fname = '.'.join(fname) + '.json'
          with open(osp.join(args.shortest_path_dir, fname), 'w') as f:
            json.dump(samples, f)

      # finished this job
      q.task_done()

  for i in range(args.num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()

  return invalid


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--question_json', default='cache/question-gen-outputs/questions_pruned_v2_with_conn.json')
  parser.add_argument('--house3d_metadata_dir', default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--suncg_data_dir', default='data/SUNCGdata')
  parser.add_argument('--graph_dir', default='cache/3d-graphs', help='directory for saving graphs')
  parser.add_argument('--target_obj_conn_map_dir', default='cache/target-obj-conn-maps')
  parser.add_argument('--target_obj_best_view_dir', default='cache/target-obj-bestview-pos')
  parser.add_argument('--colide_resolution', default=500, type=int, help='house grid resolution')
  parser.add_argument('--seed', default=24, type=int, help='random seed')
  parser.add_argument('--source_mode', default='nearby', type=str, help='nearby or random')
  parser.add_argument('--source_dist_thresh', default=30, type=int, help='max steps to source, only useful for source_mode == nearby')
  parser.add_argument('--min_dist_thresh', default=5, type=int, help='max steps to target')
  parser.add_argument('--end_dist_thresh', default=20, type=int, help='max steps to end, only useful for source_mode == nearby')
  parser.add_argument('--good_iou', default=0.25, type=float, help='a good iou for best-view point')
  parser.add_argument('--min_iou', default=0.10, type=float, help='a minimum iou for best-view point')
  parser.add_argument('--num_samples', default=5, type=int, help='number of shortest paths per question')
  parser.add_argument('--shortest_path_dir', default='cache/shortest-paths-v2', help='directory saving shortest paths for every question')
  parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
  args = parser.parse_args()

  # set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)

  # config
  cfg = {}
  cfg['colorFile'] = osp.join(args.house3d_metadata_dir, 'colormap_fine.csv')
  cfg['roomTargetFile'] = osp.join(args.house3d_metadata_dir, 'room_target_object_map.csv')
  cfg['modelCategoryFile'] = osp.join(args.house3d_metadata_dir, 'ModelCategoryMapping.csv')
  cfg['prefix'] = osp.join(args.suncg_data_dir, 'house')
  for d in cfg.values():
    assert osp.exists(d), d

  # update shortest_path_dir
  shortest_path_dir = osp.join(args.shortest_path_dir, args.source_mode+'_source_best_view')
  if not osp.isdir(shortest_path_dir):
    os.makedirs(shortest_path_dir)
  args.shortest_path_dir = shortest_path_dir
  
  # load questions = [{bbox, id, house, question, answer, type}]
  questions = json.load(open(args.question_json, 'r'))
  Questions = {qn['id']: qn for qn in questions}

  # how many unique paths are there for input questions
  visited_paths = {}
  for qn in questions:
    path = '.'.join([str(qn['house'])] + [str(box['id'])for box in qn['bbox']])
    visited_paths[path] = False
  print('There are %s unique paths for %s questions.' % (len(visited_paths), len(questions)))

  # merge questions that share same targets
  merged_questions = []
  for qn in questions:
    path = '.'.join([str(qn['house'])] + [str(box['id'])for box in qn['bbox']])
    if visited_paths[path] is False:
      merged_questions.append(qn)
      visited_paths[path] = True
  print('Correspondingly, we prepared %s merged_questions.' % len(merged_questions))
  
  # main
  house_to_qns = {}
  for qn in merged_questions:  
    # check if we've already computed this path
    path = '.'.join([str(qn['house'])] + [str(box['id'])for box in qn['bbox']])
    path_file = osp.join(args.shortest_path_dir, path+'.json')
    if osp.exists(path_file):
      print('%s computed already.' % path_file)
      continue
    house_id = qn['house']
    if house_id not in house_to_qns: house_to_qns[house_id] = []
    house_to_qns[house_id].append(qn)

  invalid = main(house_to_qns, args)

  # invalid
  invalid_json = osp.join(args.shortest_path_dir, args.source_mode+'_source_best_view_invalid.json')
  with open(invalid_json, 'w') as f:
    json.dump(invalid, f)
  print('%s invalid questions found, saved in %s' % (len(invalid), invalid_json))
  print('Done.')


  # # debug
  # qid = 88214  # 1481
  # qn = Questions[qid]
  # house_to_qns = {qn['house']: [qn]}
  # main(house_to_qns, args)


