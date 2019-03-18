import cv2
import csv
import copy
import sys
import time
import pickle
import os
import os.path as osp
import itertools
import numpy as np
from tqdm import tqdm
import pdb

import networkx as nx

from House3D.objrender import Vec3

class House3DUtils():
  def __init__(
      self,
      env,
      rotation_sensitivity=30,
      move_sensitivity=0.5,
      build_graph=False,
      graph_dir='',
      target_obj_conn_map_dir='',
      debug=True,
      load_semantic_class=True,
      collision_reward=0.0,
      success_reward=1.0,
      dist_reward_scale=1.0,
      seeing_rwd=False):
    self.env = env
    self.debug = debug

    self.rotation_sensitivity = rotation_sensitivity
    self.move_sensitivity = move_sensitivity
    self.angles = [x for x in range(-180, 180, self.rotation_sensitivity)]
    self.angle_strings = {1: 'right', -1: 'left'}
    self.dirs, self.angle_map = self.calibrate_steps(reset=True)
    self.move_multiplier = self.move_sensitivity / np.array([np.abs(x).sum() for x in self.dirs]).mean() # not sure why

    self.graph_dir = graph_dir
    self.graph = None

    self.target_obj_conn_map_dir = target_obj_conn_map_dir

    if build_graph == True:
      graph_path = osp.join(graph_dir, self.env.house.house['id']+'.pkl')
      if osp.exists(graph_path):
        self.load_graph(graph_path)
      else:
        self.build_graph(save_path=graph_path)

    self.rooms, self.objects, self.room_to_object_ids, self.object_to_room_id = self._parse() 
    self.target_id = None

    self.collision_reward = collision_reward
    self.success_reward = success_reward
    self.dist_reward_scale = dist_reward_scale
    self.seeing_rwd = seeing_rwd

    if load_semantic_class == True:
      self._load_semantic_classes() 
    

  # Shortest paths are computed in 1000 x 1000 grid coordinates.
  # One step in the SUNCG continuous coordinate system however, can be
  # multiple grids in the grid coordinate system (since turns aren't 90 deg).
  # So even though the grid shortest path is fine-grained,
  # an equivalent best-fit path in SUNCG continuous coordinates
  # has to be computed by simulating steps. Sucks, but yeah.
  #
  # For now, we first explicitly calibrate how many steps in the gridworld
  # correspond to one step in continuous world, across all directions
  def calibrate_steps(self, reset=True):
    mults, angle_map = [], {}
    cx, cy = self.env.house.to_coor(50, 50)  # ylc: probably will raise a bug, what if the robot cannot be set at (50, 50)?
    if reset == True:
      self.env.reset(x=cx, y=cy)
    for i in range(len(self.angles)):
      yaw = self.angles[i]
      self.env.cam.yaw = yaw
      self.env.cam.updateDirection()

      x1, y1 = self.env.house.to_grid(self.env.cam.pos.x, self.env.cam.pos.z)
      pos = self.env.cam.pos
      pos = pos + self.env.cam.front * self.move_sensitivity
      x2, y2 = self.env.house.to_grid(pos.x, pos.z)

      mult = np.array([x2, y2]) - np.array([x1, y1])
      mult = (mult[0], mult[1])
      angle_map[mult] = yaw
      mults.append(mult)
    return mults, angle_map


  # Go over all nodes of house environment and accumulate objects room-wise.
  def _parse(self, levelsToExplore=[0]):
    rooms, objects = {}, {}
    data = self.env.house.house

    modelCategoryMapping = {}
    csvFile = csv.reader(open(self.env.house.metaDataFile, 'r'))
    headers = next(csvFile)
    for row in csvFile:
      modelCategoryMapping[row[headers.index('model_id')]] = {
        headers[x]: row[x] for x in range(2, len(headers))  # 0 is index, 1 is model_id
      }
    for i in levelsToExplore:
      for j in range(len(data['levels'][i]['nodes'])):
        assert data['levels'][i]['nodes'][j]['type'] != 'Box'
        if 'valid' in data['levels'][i]['nodes'][j]:
          assert data['levels'][i]['nodes'][j]['valid'] == 1
        # Rooms
        if data['levels'][i]['nodes'][j]['type'] == 'Room':
          if 'roomTypes' not in data['levels'][i]['nodes'][j]:
            continue
          # Can rooms have more than one type?
          # Yes, they can; just found ['Living_Room', 'Dining_Room', 'Kitchen']
          roomType = [
            # ' '.join(x.lower().split('_'))
            x.lower() for x in data['levels'][i]['nodes'][j]['roomTypes']
          ]
          nodes = data['levels'][i]['nodes'][j]['nodeIndices'] \
            if 'nodeIndices' in data['levels'][i]['nodes'][j] else []
          rooms[data['levels'][i]['nodes'][j]['id']] = {
            'id': data['levels'][i]['nodes'][j]['id'], 
            'type': roomType,
            'bbox': data['levels'][i]['nodes'][j]['bbox'],
            'nodes': nodes,
            'model_id': data['levels'][i]['nodes'][j]['modelId']
          }
        # Objects
        elif data['levels'][i]['nodes'][j]['type'] == 'Object':
          if 'materials' not in data['levels'][i]['nodes'][j]:
            material = []
          else:
            material = data['levels'][i]['nodes'][j]['materials']
          objects[data['levels'][i]['nodes'][j]['id']] = {
            'id': data['levels'][i]['nodes'][j]['id'],
            'model_id': data['levels'][i]['nodes'][j]['modelId'],
            'fine_class': modelCategoryMapping[data['levels'][i]['nodes'][j]['modelId']]['fine_grained_class'],
            'coarse_class': modelCategoryMapping[data['levels'][i]['nodes'][j]['modelId']]['coarse_grained_class'],
            'bbox': data['levels'][i]['nodes'][j]['bbox'],
            'mat': material
          }
    room_to_object_ids, object_to_room_id = {}, {}
    for room in rooms.values():
      room_id = room['id']
      for node_idx in room['nodes']:
        node = data['levels'][0]['nodes'][node_idx]
        assert node['type'] == 'Object'
        object_id = node['id']
        room_to_object_ids[room_id] = room_to_object_ids.get(room_id, []) + [object_id]
        object_to_room_id[object_id] = room_id
    return rooms, objects, room_to_object_ids, object_to_room_id


  def _load_semantic_classes(self, color_file=None):
    if color_file == None:
      color_file = self.env.config['colorFile']

    self.semantic_classes = {}
    with open(color_file) as csv_file:
      reader = csv.DictReader(csv_file)
      for row in reader:
        c = np.array((row['r'], row['g'], row['b']), dtype=np.uint8)
        fine_cat = row['name'].lower()
        self.semantic_classes[fine_cat] = c
    return self.semantic_classes


  # takes 200-300 seconds(!) when rotation_sensitivity == 9
  def build_graph(self, save_path=None):
    # load pre-computed graph
    if osp.exists(save_path):
      self.load_graph(save_path)
      return
    # compute graph
    start_time = time.time()
    collide_res = self.env.house.n_row
    visit = dict()
    self.graph = nx.DiGraph()
    self.level_obs_map = 1 - self.env.house.levelConnMap  # 1 obstacle, 0 connected regions

    for x in range(collide_res + 1):
      for y in range(collide_res + 1):
        pos = (x, y)
        if self.env.house.canMove(x, y) and pos not in visit:
          que = [pos]
          visit[pos] = True
          ptr = 0
          while ptr < len(que):
            cx, cy = que[ptr]
            ptr += 1

            # add all angles for (cx, cy) here
            for ang in range(len(self.angles) - 1):
              self.graph.add_edge((cx, cy, self.angles[ang]), (cx, cy, self.angles[ang + 1]), weight=1.)
              self.graph.add_edge((cx, cy, self.angles[ang + 1]), (cx, cy, self.angles[ang]), weight=1.)
            # connect first and last
            self.graph.add_edge((cx, cy, self.angles[-1]), (cx, cy, self.angles[0]), weight=1.)
            self.graph.add_edge((cx, cy, self.angles[0]), (cx, cy, self.angles[-1]), weight=1.)

            for deti in range(len(self.dirs)):
              det = self.dirs[deti]
              ang = self.angle_map[det]
              tx, ty = cx + det[0], cy + det[1]
              if (self.env.house.inside(tx, ty) and
                  self.level_obs_map[min(cx, tx):max(cx, tx)+1,
                            min(cy, ty):max(cy, ty)+1].sum() == 0):
                # make changes here to add edges for angle increments as well
                #
                # cost = 1 from one angle to the next,
                # and connect first and last
                # this would be for different angles for same tx, ty
                #
                # then there would be connections for same angle
                # and from (cx, cy) to (tx, ty)
                self.graph.add_edge((cx, cy, ang), (tx, ty, ang), weight=1.)
                # licheng: also add reversed vector
                reverse_ang = ang + 180 if ang < 0 else ang - 180
                self.graph.add_edge((tx, ty, reverse_ang), (cx, cy, reverse_ang), weight=1.)
                # add to visit
                tp = (tx, ty)
                if tp not in visit:
                  visit[tp] = True
                  que.append(tp)

    print("--- %s seconds to build the graph ---" % (time.time() - start_time))
    if save_path != None:
      start_time = time.time()
      print("saving graph to %s" % (save_path))
      nx.write_gpickle(self.graph, save_path)
      print("--- %s seconds to save the graph ---" % (time.time() - start_time))

  # load pre-computed graph (~MB)
  def load_graph(self, path):
    start_time = time.time()
    self.graph = nx.read_gpickle(path)
    print("--- %s seconds to load the graph ---" % (time.time() - start_time))

  # takes 1-5 seconds when rotation_sensitivity == 9
  def compute_shortest_path(self, source, target, graph=None):
    if graph == None:
      if self.graph == None:
        graph_path = osp.join(self.graph_dir, self.env.house.house['id']+'.pkl') 
        if os.path.exists(graph_path):
          self.load_graph(graph_path)
        else:
          self.build_graph(graph_path)
      graph = self.graph

    shortest_path = nx.shortest_path(graph, source, target)
    # shortest_path = nx.dijkstra_path(graph, source, target, weight='weight')
    return shortest_path

  def _get_best_yaw_obj_from_pos(self, obj_id, grid_pos, height=1.0, use_iou=True):
    obj = self.objects[obj_id]
    obj_fine_class = obj['fine_class']

    cx, cy = self.env.house.to_coor(grid_pos[0], grid_pos[1])
    self.env.cam.pos.x = cx
    self.env.cam.pos.y = height
    self.env.cam.pos.z = cy

    best_yaw, best_coverage, best_mask = None, 0, None
    for yaw in self.angles:
      self.env.cam.yaw = yaw
      self.env.cam.updateDirection()

      seg = self.env.render(mode='semantic')
      c = self.semantic_classes[obj_fine_class.lower()]
      mask = np.all(seg == c, axis=2)
      if use_iou:
        coverage = self._compute_iou(mask)
      else:
        coverage = np.sum(mask) / (seg.shape[0] * seg.shape[1])

      if best_yaw == None:
        best_yaw = yaw
        best_coverage = coverage
        best_mask = mask
      else:
        if coverage > best_coverage:
          best_yaw = yaw
          best_coverage = coverage
          best_mask = mask

    return best_yaw, best_coverage, best_mask
  

  def _compute_iou(self, cand_mask, ref_mask=None):
    """
    Given (h, w) cand_mask, we wanna our ref_mask to be in the center of image,
    with [0.25h:0.85h, 0.25w:0.75w] occupied.
    """
    if ref_mask is None:
        h, w = cand_mask.shape[0], cand_mask.shape[1]
        ref_mask = np.zeros((h,w), np.int8)
        ref_mask[int(0.25*h):int(0.85*h), int(0.25*w):int(0.75*w)] = 1
    inter = (cand_mask > 0) & (ref_mask > 0)
    union = (cand_mask > 0) | (ref_mask > 0)
    iou = inter.sum() / (union.sum() + 1e-5)
    return iou


  def step(self, action, step_reward=False):
    """
    0: forward
    1: left
    2: right
    3: stop
    returns observation, reward, done, collision
    """
    if action not in [0, 1, 2, 3]:
      raise IndexError
    
    if step_reward == True:
      x1, y1 = self.env.house.to_grid(self.env.cam.pos.x, self.env.cam.pos.z)
      init_target_dist = self.env.house.connMap[x1, y1]  # dist to target object / dist to target room

    reward = 0
    collision = False
    done = False
    if action == 0:
      mv = self.env.move_forward(dist_fwd=self.move_sensitivity, dist_hor=0)
      obs = self.env.render()
      if not mv:  # collision
        collision = True
        reward -= self.collision_reward
      elif mv and step_reward:  # no collision and reward step
        x2, y2 = self.env.house.to_grid(self.env.cam.pos.x, self.env.cam.pos.z)
        final_target_dist = self.env.house.connMap[x2, y2]  # dist to target object / dist to target room
        reward += self.dist_reward_scale * ((init_target_dist - final_target_dist) / np.abs(
                    self.dirs[self.angles.index(self.env.cam.yaw % 180)]).sum())
    elif action == 1:  # turn left
      self.env.rotate(-self.rotation_sensitivity)
      obs = self.env.render()
    elif action == 2:  # turn right
      self.env.rotate(self.rotation_sensitivity)
      obs = self.env.render()
    else:  # stop
      done = True 
      obs = self.env.render()

    return obs, float(reward), done, collision

  def get_dist_to_target_object(self, pos):
    # pos: [x, y, z, yaw] (corr), or objrender.Vec3
    if isinstance(pos, Vec3):
      x, y = self.env.house.to_grid(pos.x, pos.z)
    else:
      x, y = self.env.house.to_grid(pos[0], pos[2])
    dist = self.env.house.connMap[x, y]
    return self.move_multiplier * dist  # why mul by move_multiplier?
  
  def get_dist_to_target_room(self, pos):
    # pos: [x, y, z, yaw] (corr), or objrender.Vec3
    if isinstance(pos, Vec3):
      x, y = self.env.house.to_grid(pos.x, pos.z)
    else:
      x, y = self.env.house.to_grid(pos[0], pos[2])
    if self.is_inside_room(pos, self.rooms[self.target_id]):
      return 0  # already inside room
    else:
      dist = self.env.house.connMap[x, y]
      return self.move_multiplier * dist
    
  def get_dist_to_target(self, pos):
    if self.target_id in self.rooms:
      return self.get_dist_to_target_room(pos)
    else:
      assert self.target_id in self.objects
      return self.get_dist_to_target_object(pos)
  
  def compute_target_iou(self, obj_id, pos=None):
    # pos: [x, y, z, yaw] (corr)
    obj = self.objects[obj_id] 
    fine_class = obj['fine_class']
    if pos:
      self.env.reset(x=pos[0], y=pos[2], yaw=pos[3])
    seg = self.env.render(mode='semantic')
    c = self.semantic_classes[fine_class.lower()]
    mask = np.all(seg == c, axis=2)
    iou = self._compute_iou(mask)
    return iou
  
  def is_inside_room(self, pos, room):
    # pos: [x, y, z, yaw] (corr), or objrender.Vec3
    if isinstance(pos, Vec3):
      x, y = pos.x, pos.z
    else:
      x, y = pos[0], pos[2]
    if x >= room['bbox']['min'][0] and x <= room['bbox']['max'][0] and \
        y >= room['bbox']['min'][2] and y <= room['bbox']['max'][2]:
      return True
    return False

####################################################################################################
# Very long connMap-related functions
####################################################################################################
  def set_target_room(self, room):
    """
    connMap computes in-house distance to the center of target room. 
    We consider center of room as target!
    """
    room_id = room['id']
    self.target_id = room_id
    connMap_file = osp.join(self.target_obj_conn_map_dir, self.env.house.house['id']+'_'+room_id+'.pkl')
    # Caching
    if room_id in self.env.house.connMapDict:
      self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = \
        self.env.house.connMapDict[room_id]
      return True
    elif osp.exists(connMap_file):
      with open(connMap_file, 'rb') as f:
        self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = \
          pickle.load(f)
      self.env.house.connMapDict[room_id] = (self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist)
      return True
    # compute connMap
    self.env.house.connMap = connMap = np.ones((self.env.house.n_row+1, self.env.house.n_row+1), dtype=np.int32) * -1
    self.env.house.inroomDist = inroomDist = np.ones((self.env.house.n_row+1, self.env.house.n_row+1), dtype=np.float32) * -1
    dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    que = []
    flag_find_open_components = True
    for _ in range(2):  # why do we need to run twice here? open + closed
      _x1, _, _y1 = room['bbox']['min']
      _x2, _, _y2 = room['bbox']['max']
      cx, cy = (_x1 + _x2) / 2, (_y1 + _y2) / 2
      x1,y1,x2,y2 = self.env.house.rescale(_x1,_y1,_x2,_y2)
      curr_components = self.env.house._find_components(x1, y1, x2, y2, dirs=dirs, return_open=flag_find_open_components)  # find all the open components
      if len(curr_components) == 0:
        print('WARNING!!!! [House] No Space Found in TargetRoom <tp=%s, bbox=[%.2f, %2f] x [%.2f, %.2f]>' %
            (targetRoomTp, _x1, _x2, _y1, _y2))
        continue
      if isinstance(curr_components[0], list):  # join all the coors in the open components
        curr_major_coors = list(itertools.chain(*curr_components))
      else:
        curr_major_coors = curr_components
      min_dist_to_center = 1e50
      for x, y in curr_major_coors:
        # connMap[x, y] = 0
        # que.append((x, y))
        tx, ty = self.env.house.to_coor(x, y)
        tdist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)  # distance in continuous space.
        if tdist < min_dist_to_center:
          min_dist_to_center = tdist
        inroomDist[x, y] = tdist
      for x, y in curr_major_coors:
        inroomDist[x, y] -= min_dist_to_center  # why subtract min_dist, isn't min_dist == 0? in case obstacle at center
      for x, y in curr_major_coors:
        if inroomDist[x, y] < .1:
          connMap[x, y] = 0
          que.append((x, y))
      if len(que) > 0: break
      if flag_find_open_components:
        # in the next loop, let's check the connectivity to closed_components
        flag_find_open_components = False
      else:
        break
      print('WARINING!!!! [House] No Space Found for Room Type {}! Now search even for closed region!!!'.format(targetRoomTp))
    assert len(que) > 0, "Error!! [House] No space found for room type {}. House ID = {}".format(targetRoomTp, (self._id if hasattr(self, '_id') else 'NA'))
    ptr = 0
    self.env.house.maxConnDist = 1
    while ptr < len(que):
      x,y = que[ptr]
      cur_dist = connMap[x, y]
      ptr += 1
      for dx,dy in dirs:
        tx,ty = x+dx,y+dy
        if self.env.house.inside(tx,ty) and self.env.house.canMove(tx,ty) and not self.env.house.isConnect(tx, ty):  # inside house, can move, not connected to target_room
          que.append((tx,ty))
          connMap[tx,ty] = cur_dist + 1  # only visiting un-connected point, earlier visited shorter path as BFS
          if cur_dist + 1 > self.env.house.maxConnDist:
            self.maxConnDist = cur_dist + 1
    self.env.house.connMapDict[room_id] = (connMap, que, inroomDist, self.env.house.maxConnDist)
    with open(connMap_file, 'wb') as f:
      pickle.dump([connMap, que, inroomDist, self.env.house.maxConnDist], f)
    self.connectedCoors = que
    print(' >>>> ConnMap Cached to %s!' % connMap_file)
    return True  # room changed!


  # analogous to `setTargetRoom` in House3D API, but difference is
  # connMap considers out-of-target region, not out-of-room region.
  # we use room/obj id as indexing, not type or name.
  def set_target_object(self, obj):
    object_id = obj['id']
    self.target_id = object_id
    room = self.rooms[self.object_to_room_id[object_id]]
    connMap_file = osp.join(self.target_obj_conn_map_dir, self.env.house.house['id']+'_'+object_id+'.pkl')
    # Caching
    if object_id in self.env.house.connMapDict:
      self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = \
        self.env.house.connMapDict[object_id]
      return True
    elif osp.exists(connMap_file):
      with open(connMap_file, 'rb') as f:
        self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = \
          pickle.load(f)
      self.env.house.connMapDict[object_id] = (self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist)
      return True
    # compute connMap
    self.env.house.connMap = connMap = np.ones((self.env.house.n_row+1, self.env.house.n_row+1), dtype=np.int32) * -1
    self.env.house.inroomDist = inroomDist = np.ones((self.env.house.n_row+1, self.env.house.n_row+1), dtype=np.float32) * -1
    dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    que = []

    _ox1, _, _oy1 = obj['bbox']['min']
    _ox2, _, _oy2 = obj['bbox']['max']
    ocx, ocy = (_ox1 + _ox2) / 2, (_oy1 + _oy2) / 2
    ox1, oy1, ox2, oy2 = self.env.house.rescale(_ox1, _oy1, _ox2, _oy2)  # target object's bounding box

    # we only need to look for components connected to outside of the room, actuallyk
    flag_find_open_components = True
    for _ in range(2):
      _x1, _, _y1 = room['bbox']['min']
      _x2, _, _y2 = room['bbox']['max']
      cx, cy = (_x1 + _x2) / 2, (_y1 + _y2) / 2
      x1, y1, x2, y2 = self.env.house.rescale(_x1, _y1, _x2, _y2)

      curr_components = self.env.house._find_components(x1, y1, x2, y2, dirs=dirs, return_open=flag_find_open_components)  # find all the open components

      if len(curr_components) == 0:
        print('No space found! =(')
        raise ValueError('no space')
      if isinstance(curr_components[0], list):  # join all the coors in the open components
        curr_major_coors = list(itertools.chain(*curr_components))
      else:
        curr_major_coors = curr_components
      
      min_dist_to_center, min_dist_to_edge = 1e50, 1e50
      for x, y in curr_major_coors:
        # Compute minimum dist to edge here
        if x in range(ox1, ox2):
          dx = 0
        elif x < ox1:
          dx = ox1 - x
        else:
          dx = x - ox2

        if y in range(oy1, oy2):
          dy = 0
        elif y < oy1:
          dy = oy1 - y
        else:
          dy = y - oy2
        
        assert dx >= 0 and dy >= 0

        if dx != 0 or dy != 0:
          dd = np.sqrt(dx**2 + dy**2)
        elif dx == 0:
          dd = dy
        else:
          dd = dx

        if dd < min_dist_to_edge:
          min_dist_to_edge = int(np.ceil(dd))
        ###
        tx, ty = self.env.house.to_coor(x, y)
        tdist = np.sqrt((tx - ocx)**2 + (ty - ocy)**2)
        if tdist < min_dist_to_center:
          min_dist_to_center = tdist
        inroomDist[x, y] = tdist
      margin = min_dist_to_edge + 1
      for x, y in curr_major_coors:
        inroomDist[x, y] -= min_dist_to_center
      for x, y in curr_major_coors:
        if x in range(ox1 - margin, ox2 + margin) and y in range(
            oy1 - margin, oy2 + margin):
          connMap[x, y] = 0
          que.append((x, y))
      if len(que) > 0: break
      if flag_find_open_components:
        flag_find_open_components = False
      else:
        break
      raise ValueError

    ptr = 0
    self.env.house.maxConnDist = 1
    while ptr < len(que):
      x, y = que[ptr]
      cur_dist = connMap[x, y]
      ptr += 1
      for dx, dy in dirs:
        tx, ty = x + dx, y + dy
        if self.env.house.inside(tx, ty) and self.env.house.canMove(
            tx, ty) and not self.env.house.isConnect(tx, ty):
          que.append((tx, ty))
          connMap[tx, ty] = cur_dist + 1
          if cur_dist + 1 > self.env.house.maxConnDist:
            self.env.house.maxConnDist = cur_dist + 1
    self.env.house.connMapDict[object_id] = (connMap, que, inroomDist, self.env.house.maxConnDist)
    with open(connMap_file, 'wb') as f:
      pickle.dump([connMap, que, inroomDist, self.env.house.maxConnDist], f)
    self.connectedCoors = que
    print(' >>>> ConnMap Cached to %s!' % connMap_file)
    return True  # room changed!

  # def set_target_room(self, room):
  #   """
  #   Slightly different from house.set_target_room, where our connMap is object-style computing 
  #   distance to the center rather than computing distance to border of room.
  #   """
  #   room_id = room['id']
  #   connMap_file = osp.join(self.target_obj_conn_map_dir, self.env.house.house['id']+'_'+room_id+'.pkl')
  #   # Caching
  #   if room_id in self.env.house.connMapDict:
  #     self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = \
  #       self.env.house.connMapDict[room_id]
  #     return True
  #   elif osp.exists(connMap_file):
  #     with open(connMap_file, 'rb') as f:
  #       self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = \
  #         pickle.load(f)
  #     self.env.house.connMapDict[room_id] = (self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist)
  #     return True
  #   # compute connMap
  #   self.env.house.connMap = connMap = np.ones((self.env.house.n_row+1, self.env.house.n_row+1), dtype=np.int32) * -1
  #   self.env.house.inroomDist = inroomDist = np.ones((self.env.house.n_row+1, self.env.house.n_row+1), dtype=np.float32) * -1
  #   dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]
  #   que = []
  #   flag_find_open_components = True
  #   for _ in range(2):  # why do we need to run twice here? open + closed
  #     _x1, _, _y1 = room['bbox']['min']
  #     _x2, _, _y2 = room['bbox']['max']
  #     cx, cy = (_x1 + _x2) / 2, (_y1 + _y2) / 2
  #     x1,y1,x2,y2 = self.env.house.rescale(_x1,_y1,_x2,_y2)
  #     curr_components = self.env.house._find_components(x1, y1, x2, y2, dirs=dirs, return_open=flag_find_open_components)  # find all the open components
  #     if len(curr_components) == 0:
  #       print('WARNING!!!! [House] No Space Found in TargetRoom <tp=%s, bbox=[%.2f, %2f] x [%.2f, %.2f]>' %
  #           (targetRoomTp, _x1, _x2, _y1, _y2))
  #       continue
  #     if isinstance(curr_components[0], list):  # join all the coors in the open components
  #       curr_major_coors = list(itertools.chain(*curr_components))
  #     else:
  #       curr_major_coors = curr_components
  #     min_dist_to_center = 1e50
  #     for x, y in curr_major_coors:
  #       # connMap[x, y] = 0
  #       # que.append((x, y))
  #       tx, ty = self.env.house.to_coor(x, y)
  #       tdist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)  # distance in continuous space.
  #       if tdist < min_dist_to_center:
  #         min_dist_to_center = tdist
  #       inroomDist[x, y] = tdist
  #     for x, y in curr_major_coors:
  #       inroomDist[x, y] -= min_dist_to_center  # why subtract min_dist, isn't min_dist == 0? in case obstacle at center
  #     for x, y in curr_major_coors:
  #       if inroomDist[x, y] < .1:
  #         connMap[x, y] = 0
  #         que.append((x, y))
  #     if len(que) > 0: break
  #     if flag_find_open_components:
  #       # in the next loop, let's check the connectivity to closed_components
  #       flag_find_open_components = False
  #     else:
  #       break
  #     print('WARINING!!!! [House] No Space Found for Room Type {}! Now search even for closed region!!!'.format(targetRoomTp))
  #   assert len(que) > 0, "Error!! [House] No space found for room type {}. House ID = {}".format(targetRoomTp, (self._id if hasattr(self, '_id') else 'NA'))
  #   ptr = 0
  #   self.env.house.maxConnDist = 1
  #   while ptr < len(que):
  #     x,y = que[ptr]
  #     cur_dist = connMap[x, y]
  #     ptr += 1
  #     for dx,dy in dirs:
  #       tx,ty = x+dx,y+dy
  #       if self.env.house.inside(tx,ty) and self.env.house.canMove(tx,ty) and not self.env.house.isConnect(tx, ty):  # inside house, can move, not connected to target_room
  #         que.append((tx,ty))
  #         connMap[tx,ty] = cur_dist + 1  # only visiting un-connected point, earlier visited shorter path as BFS
  #         if cur_dist + 1 > self.env.house.maxConnDist:
  #           self.maxConnDist = cur_dist + 1
  #   self.env.house.connMapDict[room_id] = (connMap, que, inroomDist, self.env.house.maxConnDist)
  #   with open(connMap_file, 'wb') as f:
  #     pickle.dump([connMap, que, inroomDist, self.env.house.maxConnDist], f)
  #   self.connectedCoors = que
  #   print(' >>>> ConnMap Cached to %s!' % connMap_file)
  #   return True  # room changed!

    
