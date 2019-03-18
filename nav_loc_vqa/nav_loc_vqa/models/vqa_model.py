import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ModularAttributeVQA
"""
class ModularAttributeVQA(nn.Module):
  def __init__(self, ego_feat_dim, rnn_feat_dim, cube_feat_dim, fc_dim, fc_dropout, num_answers=2):
    """
    - ego_feat_dim  : ego-centric feat dimension, i.e., (2, ego_feat_dim) or (3, ego_feat_dim)
    - rnn_feat_dim  : rnn output feat dimension
    - cube_feat_dim : cube feat dimension, i.e., (2, 4, cube_feat_dim)
    - fc_dim        : fc dimension
    - fc_dropout    : fc dropout
    - num_answers   : num. answrs (yes/no)
    """
    super(ModularAttributeVQA, self).__init__()
    self.cube_fc_layer = nn.Sequential(nn.Linear(cube_feat_dim, fc_dim), nn.ReLU())
    self.dist_fc_layer = nn.Sequential(nn.Linear(ego_feat_dim+rnn_feat_dim, fc_dim), nn.ReLU())
    self.attr_to_in = {
      'object_color_equal': 2 * rnn_feat_dim,
      'object_size_bigger': 2 * rnn_feat_dim,
      'object_size_smaller': 2 * rnn_feat_dim,
      'room_size_bigger': 2 * 4 * fc_dim,
      'room_size_smaller': 2 * 4 * fc_dim, 
      'object_dist_farther': 2 * fc_dim,  # [obj1_feats * obj2_feats, obj2_feats * obj3_feats]
      'object_dist_closer': 2 * fc_dim,
    }
    self.attr_mlp = {}
    for attr in self.attr_to_in.keys():
      num_input = self.attr_to_in[attr]
      mod = nn.Sequential(
              nn.Linear(self.attr_to_in[attr], fc_dim),
              nn.Dropout(fc_dropout),
              nn.ReLU(),
              nn.Linear(fc_dim, num_answers),
            )
      self.add_module(attr, mod)
      self.attr_mlp[attr] = mod

  def forward(self, img_feats_list, attrs):
    """
    Inputs:
    - img_feats_list : list of ego_feats (2 or 3, ego_feat_dim) or cube_feats (2, 4, cube_feat_dim)
    - attrs          : n attr types, used to choose sub-network
    Outputs:
    - scores         : (n, num_answers) before softmax
    """
    n = len(img_feats_list)
    scores = []
    for i in range(n):
      img_feats = img_feats_list[i]
      attr = attrs[i]
      if 'room_size' in attr:
        # (2, 4, cube_feat_dim) --> (2, 4, fc_dim)
        fc_feats = self.cube_fc_layer(img_feats) 
      else:
        if 'object_dist' in attr:
          # (3, ego_feat_dim+rnn_feat_dim) --> (3, fc_dim)
          assert img_feats.shape[0] == 3
          fc_feats = self.dist_fc_layer(img_feats)  
          fc_feats = torch.cat([fc_feats[0:1] * fc_feats[1:2], fc_feats[1:2] * fc_feats[2:3]], 0) 
        else:
          # (2, ego_feat_dim)
          fc_feats = img_feats  
      # flatten fc_feats
      fc_feats = fc_feats.view(1, -1)       
      score = self.attr_mlp[attr](fc_feats) # (1, num_answers)
      scores.append(score)
    # return
    scores = torch.cat(scores, 0)  # (n, num_answers)
    return scores

