# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize_Scale(nn.Module):
  def __init__(self, dim, init_norm=20):
    super(Normalize_Scale, self).__init__()
    self.init_norm = init_norm
    self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

  def forward(self, bottom):
    # input is variable (n, dim)
    bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
    bottom_normalized_scaled = bottom_normalized * self.weight
    return bottom_normalized_scaled

"""
SelectMaxMarginCriterion
"""
class SelectMaxMarginCriterion(nn.Module):
  def __init__(self, margin):
    super(SelectMaxMarginCriterion, self).__init__()
    self.margin = margin
  
  def forward(self, sc0, sc1, masks):
    """
    Inputs:
    - sc0   : (n, ) float
    - sc1   : (n, ) float
    - masks : (n, ) list, where 1 computs max(m + sc0 - sc1, 0) and 0 computs max(m + sc1 - sc0, 0)
    indicating 1 wants sc1 bigger, and 0 wants sc0 bigger
    """
    batch_size = sc0.shape[0] 
    loss0 = torch.clamp(self.margin + sc1 - sc0, min=0)  # we wanna sc0 > sc1
    loss1 = torch.clamp(self.margin + sc0 - sc1, min=0)  # we wanna sc1 > sc0
    losses = torch.cat([loss0.view(-1, 1), loss1.view(-1, 1)], 1)  # (n, 2)
    loss = losses.gather(1, torch.LongTensor(masks).view(-1, 1).cuda()) # (n, 1)
    return loss.sum() / batch_size

"""
Used for object distance comparison 
"""
class CompareObjectDist(nn.Module):
  def __init__(self):
    super(CompareObjectDist, self).__init__()
    self.mlp = nn.Sequential(nn.Linear(3200, 64), nn.BatchNorm1d(64), )
  
  def cossim(self, f1, f2):
    # f1 (n, d), f2(n, d) --> cossim (n, 1)
    f1 = F.normalize(f1, p=2, dim=1)  # (n, d)
    f2 = F.normalize(f2, p=2, dim=1)  # (n, d)
    cossim = torch.sum(f1 * f2, 1)  # (n, )
    return cossim
  
  def forward(self, obj1_conv4_feats, obj2_conv4_feats, obj3_conv4_feats):
    """
    Inputs:
    - obj1_conv4_feats (n, 32, 10, 10)
    - obj2_conv4_feats (n, 32, 10, 10)
    - obj3_conv4_feats (n, 32, 10, 10)
    Output:
    - cossim12 (n, )
    - cossim23 (n, )
    """
    n = obj1_conv4_feats.shape[0]
    obj1_feats = self.mlp(obj1_conv4_feats.reshape(n, -1))  # (n, d)
    obj2_feats = self.mlp(obj2_conv4_feats.reshape(n, -1))  # (n, d)
    obj3_feats = self.mlp(obj3_conv4_feats.reshape(n, -1))  # (n, d)
    # compute cossim12, cossim23
    cossim12 = self.cossim(obj1_feats, obj2_feats)  # (n, )
    cossim23 = self.cossim(obj2_feats, obj3_feats)  # (n, )
    return cossim12, cossim23

  def inference(self, obj1_conv4_feats, obj2_conv4_feats, obj3_conv4_feats, ops):
    """
    Inputs:
    - obj1_conv4_feats, obj2_conv4_feats, obj3_conv4_feats: (n, 32, 10, 10)
    - ops: closer or farther
    Return:
    - preds    : list of [1 (yes) / 0 (no)]
    - cossim12 : (n, )
    - cossim23 : (n, )
    """
    n = obj1_conv4_feats.shape[0]
    obj1_feats = self.mlp(obj1_conv4_feats.reshape(n, -1))  # (n, d)
    obj2_feats = self.mlp(obj2_conv4_feats.reshape(n, -1))  # (n, d)
    obj3_feats = self.mlp(obj3_conv4_feats.reshape(n, -1))  # (n, d)
    # compute cossim12, cossim23
    cossim12 = self.cossim(obj1_feats, obj2_feats)  # (n, )
    cossim23 = self.cossim(obj2_feats, obj3_feats)  # (n, )
    # inference
    preds = []
    assert n == len(ops)
    for i, op in enumerate(ops):
      assert op in ['closer', 'farther']
      if op == 'closer':
        if cossim12[i] > cossim23[i]:
          preds.append(1)
        else:
          preds.append(0)
      else:
        if cossim12[i] > cossim23[i]:
          preds.append(0)
        else:
          preds.append(1)   
    return preds, cossim12, cossim23

"""
Concat two (segm, depth, segm x depth^2) feats, and [bigger/smaller] gate,
computing their equality score [0, 1].
"""
class CompareObjectSize(nn.Module):
  def __init__(self, depth_square):
    super(CompareObjectSize, self).__init__()
    self.fc = nn.Linear(3136, 1, bias=False)
    self.fc.weight.data.fill_(1/3136)
    self.depth_square = depth_square > 0

  def forward(self, segm1, depth1, segm2, depth2):
    """
    Inputs:
    - segm1  (n, 224, 224) float {0., 1.} score
    - depth1 (n, 224, 224) float [0., 1.] score (after sigmoid)
    - segm2  (n, 224, 224) float {0., 1.} score
    - depth2 (n, 224, 224) float [0., 1.] score (after sigmoid)
    Output:
    - sc1 (n, )
    - sc2 (n, )
    """
    n = segm1.shape[0]
    masked_depth1 = segm1 * depth1**2 if self.depth_square else segm1 * depth1 # (n, 224, 224)
    masked_depth1 = F.max_pool2d(masked_depth1.unsqueeze(1), kernel_size=4, stride=4)  # (n, 1, 56, 56)
    masked_depth1 = masked_depth1.reshape(n, -1)  # (n, 3136)
    sc1 = self.fc(masked_depth1)   # (n, 1)

    masked_depth2 = segm2 * depth2**2 if self.depth_square else segm2 * depth2 # (n, 224, 224)
    masked_depth2 = F.max_pool2d(masked_depth2.unsqueeze(1), kernel_size=4, stride=4)  # (n, 1, 56, 56)
    masked_depth2 = masked_depth2.reshape(n, -1)  # (n, 3136)
    sc2 = self.fc(masked_depth2)   # (n, 1)

    return sc1.view(-1), sc2.view(-1)

  def inference(self, segm1, depth1, segm2, depth2, ops):
    """
    Inputs:
    - segm1  : (n, 224, 224) float {0., 1.} score
    - depth1 : (n, 224, 224) float [0., 1.] score (after sigmoid)
    - segm2  : (n, 224, 224) float {0., 1.} score
    - depth2 : (n, 224, 224) float [0., 1.] score (after sigmoid)
    - ops    : list of [bigger / smaller]
    Return:
    - preds  : list of [1 (yes) / 0 (no)]
    - sc1    : size score of 1
    - sc2    : size score of 2
    """
    n = segm1.shape[0]
    masked_depth1 = segm1 * depth1**2 if self.depth_square else segm1 * depth1 # (n, 224, 224)
    masked_depth1 = F.max_pool2d(masked_depth1.unsqueeze(1), kernel_size=4, stride=4)  # (n, 1, 56, 56)
    masked_depth1 = masked_depth1.reshape(n, -1)  # (n, 3136)
    sc1 = self.fc(masked_depth1)  # (n, 1)

    masked_depth2 = segm2 * depth2**2 if self.depth_square else segm2 * depth2 # (n, 224, 224)
    masked_depth2 = F.max_pool2d(masked_depth2.unsqueeze(1), kernel_size=4, stride=4)  # (n, 1, 56, 56)
    masked_depth2 = masked_depth2.reshape(n, -1)  # (n, 3136)
    sc2 = self.fc(masked_depth2)  # (n, 1)

    # inference
    preds = []
    assert n == len(ops)
    for i, op in enumerate(ops):
      assert op in ['bigger', 'smaller']
      if op == 'bigger':
        if sc1[i] > sc2[i]:
          preds.append(1)
        else:
          preds.append(0)
      else:
        if sc1[i] > sc2[i]:
          preds.append(0)
        else:
          preds.append(1)   
    return preds, sc1, sc2

"""
Used for room size comparison.
The inputs are:
- room1_cube_depth 
"""
class CompareRoomSize(nn.Module):
  def __init__(self, depth_square):
    super(CompareRoomSize, self).__init__()
    self.fc = nn.Linear(3136, 1, bias=False)
    self.fc.weight.data.fill_(1/3136)
    self.depth_square = depth_square > 0

  def forward(self, room1_cube_depth, room2_cube_depth):
    """
    Inputs:
    - room1_cube_depth (n, 4, 224, 224)
    - room2_cube_depth (n, 4, 224, 224)
    - ops    [bigger, smaller, ...]
    Output
    - sc1 (n, )
    - sc2 (n, )
    """
    n = room1_cube_depth.shape[0]
    room1_cube_depth = F.max_pool2d(room1_cube_depth, kernel_size=8, stride=8)  # (n, 4, 28, 28)
    room1_cube_depth = room1_cube_depth.view(n, -1)  # (n, 3136) 
    if self.depth_square: room1_cube_depth = room1_cube_depth ** 2
    sc1 = self.fc(room1_cube_depth)

    room2_cube_depth = F.max_pool2d(room2_cube_depth, kernel_size=8, stride=8)  # (n, 4, 28, 28)
    room2_cube_depth = room2_cube_depth.view(n, -1)  # (n, 3136)
    if self.depth_square: room2_cube_depth = room2_cube_depth ** 2
    sc2 = self.fc(room2_cube_depth)

    return sc1.view(-1), sc2.view(-1)
    
  def inference(self, room1_cube_depth, room2_cube_depth, ops):
    """
    Inputs:
    - room1_cube_depth (n, 4, 224, 224)
    - room2_cube_depth (n, 4, 224, 224)
    - ops [bigger, smaller, ...]
    Return:
    - preds : list of [1(yes) / 0(no)]
    - sc1   : size score of room1
    - sc2   : size score of room2
    """
    n = room1_cube_depth.shape[0]
    room1_cube_depth = F.max_pool2d(room1_cube_depth, kernel_size=8, stride=8)  # (n, 4, 28, 28)
    room1_cube_depth = room1_cube_depth.view(n, -1)  # (n, 3136) 
    if self.depth_square: room1_cube_depth = room1_cube_depth ** 2
    sc1 = self.fc(room1_cube_depth)

    room2_cube_depth = F.max_pool2d(room2_cube_depth, kernel_size=8, stride=8)  # (n, 4, 28, 28)
    room2_cube_depth = room2_cube_depth.view(n, -1)  # (n, 3136)
    if self.depth_square: room2_cube_depth = room2_cube_depth ** 2
    sc2 = self.fc(room2_cube_depth)

    # inference
    preds = []
    assert n == len(ops)
    for i, op in enumerate(ops):
      assert op in ['bigger', 'smaller']
      if op == 'bigger':
        if sc1[i] > sc2[i]:
          preds.append(1)
        else:
          preds.append(0)
      else:
        if sc1[i] > sc2[i]:
          preds.append(0)
        else:
          preds.append(1)   
    return preds, sc1, sc2

"""
Concat feats1 and feats2, computing their equalty score [0, 1].
"""
class EqualObjectColor(nn.Module):
  def __init__(self, input_dim, fc_dim, drop_out=0.2):
    super(EqualObjectColor, self).__init__()
    self.mlp = nn.Sequential(
          nn.Linear(input_dim*2, fc_dim),
          nn.ReLU(),
          nn.Dropout(drop_out),
          nn.Linear(fc_dim, 1))

  def forward(self, feats1, feats2):
    """
    Inputs:
    - feats1 (n, d)
    - feats2 (n, d)
    Output:
    - metric score: (n, ) between 0 and 1
    Note the input order.
    """
    feats = torch.cat([feats1, feats2], 1)  # (n, 2d)
    score = self.mlp(feats)
    score = score.view(-1)
    score = F.sigmoid(score)
    return score
"""
Used for phrase-guided color prediction
1) For color prediction, the inputs are:
- phrase (n, 300)
- feats  (n, 512, 6, 6)
Output is 
- weighted_feats: (n, 512)
- attn   (n, 36)
- color_scores (n, #colors)
"""
class PhraseGuidedColor(nn.Module):
  def __init__(self, num_colors, wordvec_dim=300, feats_dim=512, jemb_dim=256, 
               phrase_init_norm=20, visual_init_norm=20):
    super(PhraseGuidedColor, self).__init__()
    self.wordvec_dim = wordvec_dim
    self.feats_dim = feats_dim
    self.jemb_dim = jemb_dim
    # object phrase embedding
    self.phrase_embed = nn.Sequential(nn.Linear(wordvec_dim, jemb_dim),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))
    self.phrase_normalizer = Normalize_Scale(jemb_dim, phrase_init_norm)
    # color attention fusion layer
    self.conv5_normalizer = Normalize_Scale(feats_dim, visual_init_norm)
    self.color_attn_fuse = nn.Sequential(nn.Linear(feats_dim + jemb_dim, jemb_dim),
                                         nn.Tanh(),
                                         nn.Linear(jemb_dim, 1))
    # color fc layer
    self.color_fc = nn.Linear(feats_dim, num_colors)

  def forward(self, obj_phrase, conv5_feats):
    """
    Inputs:
    - obj_phrase     (n, 300)
    - conv5_feats    (n, 512, 6, 6)
    Output:
    - weighted_feats (n, 512)
    - attn           (n, 36)
    - color_scores   (n, #colors)
    """
    # phrase_feats
    bz = obj_phrase.shape[0]
    phrase_feats = self.phrase_normalizer(self.phrase_embed(obj_phrase))  # (n, jemb_dim)
    phrase_feats = phrase_feats.unsqueeze(1).expand(bz, 36, self.jemb_dim)  # (n, 36, jemb_dim)
    phrase_feats = phrase_feats.contiguous().view(-1, self.jemb_dim)  # (n*36, jemb_dim)

    # visual_feats
    visual_feats = conv5_feats.view(bz, 512, -1)  # (n, 512, 36)
    visual_feats = visual_feats.transpose(1,2).contiguous().view(-1, 512)  # (n*36, 512)
    visual_feats = self.conv5_normalizer(visual_feats)  # (n*36, 512)

    # compute spatial attention
    fused_feats = torch.cat([visual_feats, phrase_feats], 1)  # (n*36, 512+512)
    attn = self.color_attn_fuse(fused_feats)  # (n*36, 1)
    attn = F.softmax(attn.view(bz, 36), 1)  # (n, 36)

    # weighted sum
    attn3 = attn.unsqueeze(1)  # (n, 1, 36)
    weighted_feats = torch.bmm(attn3, visual_feats.view(bz, 36, -1))  # (n, 1, 512)
    weighted_feats = weighted_feats.squeeze(1)  # (n, 512)

    # color predictions
    color_scores = self.color_fc(weighted_feats)  # (n, #colors)

    return weighted_feats, attn, color_scores
"""
Used for phrase-guided object segmentation
For object segmentation, the inputs are:
- phrase (n, 300)
- segm_feats  (n, 191, H, W)
Output is 
- segm_scores (n, 224, 224)
"""
class PhraseGuidedSegmentation(nn.Module):
  def __init__(self, wordvec_dim=300, seg_dim=191, jemb_dim=256):
    super(PhraseGuidedSegmentation, self).__init__()
    self.wordvec_dim = wordvec_dim
    self.seg_dim = seg_dim
    self.jemb_dim = jemb_dim
    # nets
    self.obj_embed = nn.Sequential(nn.Linear(wordvec_dim, jemb_dim), 
                                   nn.BatchNorm1d(jemb_dim),
                                   nn.ReLU(), 
                                   nn.Dropout(0.5))
    self.attn_fuse = nn.Sequential(nn.Conv2d(seg_dim+jemb_dim, jemb_dim, 1),
                                   nn.BatchNorm2d(jemb_dim),
                                   nn.ReLU(),
                                   nn.Conv2d(jemb_dim, 1, 1))

  def forward(self, obj_phrase, segm_feats):
    """
    Inputs:
    - segm_feats         (n, 191, H, W)
    - obj_phrase         (n, 300)
    Output:
    - phrase_guided_segm (n, 224, 224) before sigmoid
    """
    # expand obj_phrase
    batch, seg_dim, H, W = segm_feats.size(0), segm_feats.size(1), segm_feats.size(2), segm_feats.size(3)  # n, 191, H, W
    
    embedding = self.obj_embed(obj_phrase)  # (n, jemb_dim)
    embedding = embedding.unsqueeze(2).unsqueeze(3)  # (n, jemb_dim, 1, 1)
    embedding = embedding.expand(batch, self.jemb_dim, H, W)  # (n, jemb_dim, H, W)

    # concat obj_phrase and segm_feats
    embed_segm_feats = torch.cat([segm_feats, embedding], 1)  # (n, jemb_dim + seg_dim, H, W)
    embed_segm_score = self.attn_fuse(embed_segm_feats)  # (n, 1, H, W)
    segm_out = F.upsample(embed_segm_score, (224, 224), mode='bilinear', align_corners=True)  # (n, 1, 224, 224)
    segm_out = segm_out.squeeze(1)  # (n, 224, 224)
    return segm_out

#########################################################
# Modular Attribute Network
#########################################################
class ModularAttributeVQA(nn.Module):
  def __init__(self, img_feat_dim, fc_dim, fc_dropout, use_cube=False, num_answers=2):
    """
    - img_feat_dim: input image feats dimension
    - fc_dim      : fc dimension
    - fc_dropout  : fc dropout
    - num_answers : num. answers (yes/no)
    """
    super(ModularAttributeVQA, self).__init__()
    self.use_cube = use_cube
    self.attr_to_in = {
      'object_color_equal': 2,
      'object_size_bigger': 2,
      'object_size_smaller': 2,
      'room_size_bigger': 2*4 if self.use_cube else 2,   # [cube_img1, cube_img2]
      'room_size_smaller': 2*4 if self.use_cube else 2,
      'object_dist_farther': 2,  # [obj1_feats * obj2_feats, obj2_feats * obj3_feats]
      'object_dist_closer': 2,
    }

    # attribute-based visual encoder
    self.cnn_fc_layer = nn.Sequential(nn.Linear(img_feat_dim, fc_dim), nn.ReLU(),)
    self.attr_mlp = {}
    for attr in self.attr_to_in.keys():
      num_input = self.attr_to_in[attr]
      mod = nn.Sequential(nn.Linear(self.attr_to_in[attr]*fc_dim, fc_dim), 
                          nn.Dropout(fc_dropout),
                          nn.ReLU(),
                          nn.Linear(fc_dim, num_answers))
      self.add_module(attr, mod)
      self.attr_mlp[attr] = mod

  def forward(self, img_feats, attrs):
    """
    Inputs:
    - img_feats  : (n, max_num_inputs(8), 3200)
    - attrs      : n attribute types, used to filter img_feats (by number of inputs)
    Outputs:
    - scores     : (n, num_answers)
    """
    img_feats = self.cnn_fc_layer(img_feats)  # (n, max_num_inputs, fc_dim)
    scores = []
    for i in range(img_feats.shape[0]):
      attr = attrs[i]
      if 'object_dist' in attr:
        img_feat1 = img_feats[i, 0:1] # (1, fc_dim)
        img_feat2 = img_feats[i, 1:2] # (1, fc_dim)
        img_feat3 = img_feats[i, 2:3] # (1, fc_dim)
        filtered_img_feats = torch.cat([img_feat1 * img_feat2, img_feat2 * img_feat3], 0)  # (2, fc_dim)
        # filtered_img_feats = torch.cat([img_feat2 - img_feat1, img_feat2 - img_feat3], 0)  # (2, fc_dim)
        filtered_img_feats = filtered_img_feats.view(1, -1)  # (1, 2*fc_dim)
      else:
        num_in = self.attr_to_in[attr]
        filtered_img_feats = img_feats[i, :num_in] # (1, num_inputs, fc_dim)
        filtered_img_feats = filtered_img_feats.view(1, -1) # (1, num_inputs * fc_dim)
      # forward
      score = self.attr_mlp[attr](filtered_img_feats)  # (1, num_answers)
      scores.append(score)
    scores = torch.cat(scores, 0)  # (n, num_answers)
    return scores

# """
# Used for 1) phrase-guided color prediction, 2) phrase-guided object segmentation
# 1) For color prediction, the inputs are:
# - phrase (n, 300)
# - feats  (n, 512, 6, 6)
# Output is 
# - phrase_weighted_feats: (n, 512)
# - attn   (n, 36)
# - color_scores (n, #colors)
# 2) For object segmentation, the inputs are:
# - phrase (n, 300)
# - feats  (n, 191, 24, 24)
# Output is 
# - segm   (n, 224, 224)
# """
# class ObjectAttention(nn.Module):
#   def __init__(self, num_colors, wordvec_dim=300, feats_dim=512, segm_dim=191, jemb_dim=256, 
#                phrase_init_norm=20, visual_init_norm=20):
#     super(ObjectAttention, self).__init__()
#     self.wordvec_dim = wordvec_dim
#     self.feats_dim = feats_dim
#     self.segm_dim = segm_dim
#     self.jemb_dim = jemb_dim
#     # object phrase embedding
#     self.phrase_embed = nn.Sequential(nn.Linear(wordvec_dim, jemb_dim),
#                                       nn.ReLU(),
#                                       nn.Dropout(0.5))
#     self.phrase_normalizer_c = Normalize_Scale(jemb_dim, phrase_init_norm)  # used for color
#     self.phrase_normalizer_s = Normalize_Scale(jemb_dim, phrase_init_norm)  # used for segm
#     # color attention fusion layer
#     self.conv5_normalizer = Normalize_Scale(feats_dim, visual_init_norm)  # used for conv5_feats
#     self.color_attn_fuse = nn.Sequential(nn.Linear(feats_dim + jemb_dim, jemb_dim),
#                                          nn.Tanh(),
#                                          nn.Linear(jemb_dim, 1))
#     # color fc layer
#     self.color_fc = nn.Linear(feats_dim, num_colors)
#     # object segm fusion layer
#     self.segm_normalizer = Normalize_Scale(segm_dim, visual_init_norm)  # used for segm_feats (n, 191, 24, 24)
#     self.segm_attn_fuse = nn.Sequential(nn.Linear(segm_dim + jemb_dim, jemb_dim),
#                                         nn.Tanh(),
#                                         nn.Linear(jemb_dim, 1))

#   def phrase_guided_color(self, obj_phrase, conv5_feats):
#     """
#     Inputs:
#     - obj_phrase     (n, 300)
#     - conv5_feats    (n, 512, 6, 6)
#     Output:
#     - weighted_feats (n, 512)
#     - attn           (n, 6, 6)
#     - color_scores   (n, #colors)
#     """
#     # phrase_feats
#     bz = obj_phrase.shape[0]
#     phrase_feats = self.phrase_normalizer_c(self.phrase_embed(obj_phrase))  # (n, jemb_dim)
#     phrase_feats = phrase_feats.unsqueeze(1).expand(bz, 36, self.jemb_dim)  # (n, 36, jemb_dim)
#     phrase_feats = phrase_feats.contiguous().view(-1, self.jemb_dim)  # (n*36, jemb_dim)

#     # visual_feats
#     visual_feats = conv5_feats.view(bz, 512, -1)  # (n, 512, 36)
#     visual_feats = visual_feats.transpose(1,2).contiguous().view(-1, 512)  # (n*36, 512)
#     visual_feats = self.conv5_normalizer(visual_feats)  # (n*36, 512)

#     # compute spatial attention
#     fused_feats = torch.cat([visual_feats, phrase_feats], 1)  # (n*36, 512+512)
#     attn = self.color_attn_fuse(fused_feats)  # (n*36, 1)
#     attn = F.softmax(attn.view(bz, 36), 1)  # (n, 36)

#     # weighted sum
#     attn3 = attn.unsqueeze(1)  # (n, 1, 36)
#     weighted_feats = torch.bmm(attn3, visual_feats.view(bz, 36, -1))  # (n, 1, 512)
#     weighted_feats = weighted_feats.squeeze(1)  # (n, 512)

#     # color predictions
#     color_scores = self.color_fc(weighted_feats)  # (n, #colors)

#     return weighted_feats, attn, color_scores

#   def phrase_guided_segm(self, obj_phrase, segm_feats):
#     """
#     Inputs:
#     - obj_phrase  (n, 300)
#     - segm_feats  (n, 191, 6, 6)
#     Output:
#     - segm_scores (n, 224, 224)
#     """
#     # phrase_feats
#     bz = obj_phrase.shape[0]
#     phrase_feats = self.phrase_normalizer_s(self.phrase_embed(obj_phrase))  # (n, jemb_dim)
#     phrase_feats = phrase_feats.unsqueeze(1).expand(bz, 36, self.jemb_dim)  # (n, 36, jemb_dim)
#     phrase_feats = phrase_feats.contiguous().view(-1, self.jemb_dim)  # (n*36, jemb_dim)

#     # visual_feats
#     visual_feats = segm_feats.view(bz, 191, -1)  # (n, 191, 36)
#     visual_feats = visual_feats.transpose(1, 2).contiguous().view(-1, 191)  # (n*36, 191)
#     visual_feats = self.segm_normalizer(visual_feats)  # (n*36, 191)

#     # compute spatial attention (segmentation)
#     fused_feats = torch.cat([visual_feats, phrase_feats], 1)  # (n*36, jemb_dim + 191)
#     attn = self.segm_attn_fuse(fused_feats)   # (n * 36, 1)
#     attn = attn.view(bz, -1).view(bz, 1, 24, 24)  # (n, 1, 6, 6)
#     segm_scores = F.upsample(attn, (224, 224), mode='bilinear', align_corners=True)  # (n, 1, 224, 224)
#     return segm_scores



# if __name__ == '__main__':

#   obj_attn_layer = ObjectAttention(num_colors=3)
#   n = 4
#   conv5_feats = torch.randn(n, 512, 6, 6)
#   obj_phrase = torch.randn(n, 300)
#   weighted_feats, attn, color_scores = obj_attn_layer.phrase_guided_color(obj_phrase, conv5_feats)
#   print(weighted_feats.shape, attn.shape, color_scores.shape)

#   segm_feats = torch.randn(n, 191, 24, 24)
#   obj_phrase = torch.randn(n, 300)
#   segm_out = obj_attn_layer.phrase_guided_segm(obj_phrase, segm_feats)
#   print(segm_out.shape)


#########################################################
# some old functions
#########################################################
"""
MaxMarginCriterion
"""
class MaxMarginCriterion(nn.Module):
  def __init__(self, margin):
    super(MaxMarginCriterion, self).__init__()
    self.margin = margin
  
  def forward(self, cossim12, cossim23):
    """
    Inputs:
    - cossim12 : (n, )
    - cossim23 : (n, )
    Output:
    - max_margin_loss = max(margin + cossim23 - cossim12, 0)
    """
    batch_size = cossim12.size(0)
    loss = torch.clamp(self.margin + cossim23 - cossim12, min=0)
    return loss.sum() / batch_size

"""
Used for object distance comparison 
"""
class EqualObjectDist(nn.Module):
  def __init__(self):
    super(EqualObjectDist, self).__init__()
    self.mlp = nn.Sequential(nn.Linear(3200, 64), nn.BatchNorm1d(64), )
    self.fc = nn.Sequential(nn.Linear(2, 2))
  
  def forward(self, obj1_conv4_feats, obj2_conv4_feats, obj3_conv4_feats, ops):
    """
    Inputs:
    - obj1_conv4_feats (n, 32, 10, 10)
    - obj2_conv4_feats (n, 32, 10, 10)
    - obj3_conv4_feats (n, 32, 10, 10)
    - ops
    Output:
    - scores (n, ) between 0 and 1 
    """
    def cossim(f1, f2):
      # f1 (n, d), f2(n, d) --> cossim (n, 1)
      f1 = F.normalize(f1, p=2, dim=1)  # (n, d)
      f2 = F.normalize(f2, p=2, dim=1)  # (n, d)
      cossim = torch.sum(f1 * f2, 1)  # (n, )
      return cossim.view(-1, 1)

    for op in ops:
      assert op in ['closer', 'farther']
    n = obj1_conv4_feats.shape[0]
    obj1_feats = self.mlp(obj1_conv4_feats.reshape(n, -1))  # (n, d)
    obj2_feats = self.mlp(obj2_conv4_feats.reshape(n, -1))  # (n, d)
    obj3_feats = self.mlp(obj3_conv4_feats.reshape(n, -1))  # (n, d)
    # compute cossim12, cossim23
    cossim12 = cossim(obj1_feats, obj2_feats)  # (n, 1)
    cossim23 = cossim(obj2_feats, obj3_feats)  # (n, 1)
    # compute comparison score (n, 2)
    scores = self.fc(torch.cat([cossim12, cossim23], 1))  # (n, 2)
    scores = F.softmax(scores, dim=1)  # (n, 2)
    masks = [0 if op == 'closer' else 1 for op in ops]  # (n, )
    scores = scores.gather(1, torch.LongTensor(masks).view(-1, 1).cuda()) # (n, 1)
    scores = scores.view(-1)  # (n, )
    return scores

"""
Used for room size comparison.
The inputs are:
- room1_cube_depth 
"""
class EqualRoomSize(nn.Module):
  def __init__(self, input_dim=8, fc_dim=32, dropout=0.1):
    super(EqualRoomSize, self).__init__()
    self.mlp = nn.Sequential(
                  nn.Linear(input_dim*3, fc_dim), 
                  nn.BatchNorm1d(fc_dim), 
                  nn.ReLU(), 
                  nn.Dropout(dropout),
                  nn.Linear(fc_dim, 2))

  def extract_feats(self, cube_depth):
    """
    Input:
    - room_cube_depth : (n, 4, 224, 224) float
    Output:
    - feats: (n, 8) which is 
      [d0.mean, d1.mean(), d2.mean(), d3.mean(), (d0.mean()+d2.mean())*(d1.mean()+d3.mean()),
       d0.mean()*d2.mean(), d1.mean()*d3.mean(), d0.mean()*d1.mean()*d2.mean()*d3.mean()]
    """
    feats = cube_depth.data.new(cube_depth.size(0), 8).zero_()
    dmeans = cube_depth.mean(3).mean(2)  # (n, 4)
    d0, d1, d2, d3 = dmeans[:, 0], dmeans[:, 1], dmeans[:, 2], dmeans[:, 3]  # (n, ) each
    feats[:, :4] = dmeans
    feats[:, 4] = (d0+d2) * (d1+d3)
    feats[:, 5] = d0 * d2
    feats[:, 6] = d1 * d3
    feats[:, 7] = d0 * d1 * d2 * d3
    return feats

  def forward(self, room1_cube_depth, room2_cube_depth, ops):
    """
    Inputs:
    - room1_cube_depth (n, 4, 224, 224)
    - room2_cube_depth (n, 4, 224, 224)
    - ops    [bigger, smaller, ...]
    Output
    - metric score: (n, ) between 0 and 1
    Note the input feats order as we are computing feats1[op]feats2
    """
    feats1 = self.extract_feats(room1_cube_depth)  # (n, 8)
    feats2 = self.extract_feats(room2_cube_depth)  # (n, 8)
    feats = torch.cat([feats1, feats2, feats1-feats2], 1)  # (n, 8x3)
    scores = self.mlp(feats)  # (n, 2)
    masks = [1 if op == 'bigger' else 0 for op in ops]  # (n, )
    scores = scores.gather(1, torch.LongTensor(masks).view(-1, 1).cuda())  # (n, 1)
    scores = F.sigmoid(scores).view(-1)  # (n, )
    return scores
"""
Concat two (segm, depth, segm x depth^2) feats, and [bigger/smaller] gate,
computing their equality score [0, 1].
"""
class EqualObjectSize(nn.Module):
  def __init__(self, input_dim=1, fc_dim=5, dropout=0.1):
    super(EqualObjectSize, self).__init__()
    self.mlp = nn.Sequential(nn.Linear(input_dim*3, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, 2))
  
  def estimate_normalized_size(self, segm, depth):
    """
    Inputs:
    - segm  (n, 224, 224) float {0, 1}
    - depth (n, 224, 224) float {0, 1}
    Output:
    - obj_size: (n, d)
    """
    masked_depth = segm * depth  # (n, 224, 224)
    obj_area = segm.sum(2).sum(1) / (224*224)  # (n, )
    obj_depth = masked_depth.sum(2).sum(1) / segm.sum(2).sum(1)  # (n, )
    obj_size = segm.sum(2).sum(1) * (obj_depth**2)  # (n, )
    obj_size = obj_size.view(-1, 1)
    # feats = torch.cat([obj_area.view(-1, 1), obj_depth.view(-1, 1), obj_size.view(-1,1)/224.], 1)  # (n, 3)
    feats = obj_size
    return feats 

  def forward(self, segm1, depth1, segm2, depth2, ops):
    """
    Inputs:
    - segm1  (n, 224, 224) float {0., 1.} score
    - depth1 (n, 224, 224) float [0., 1.] score (after sigmoid)
    - segm2  (n, 224, 224) float {0., 1.} score
    - depth2 (n, 224, 224) float [0., 1.] score (after sigmoid)
    - ops    list of n ['bigger'/'smaller']
    Output:
    - metric scores: (n, )  between 0 and 1
    """
    feats1 = self.estimate_normalized_size(segm1, depth1)  # (n, d)
    feats2 = self.estimate_normalized_size(segm2, depth2)  # (n, d)
    feats = torch.cat([feats1, feats2, feats1-feats2], 1)  # (n, 3d)
    scores = self.mlp(feats)  # (n, 2)
    masks = [1 if op == 'bigger' else 0 for op in ops]  # (n, )
    scores = scores.gather(1, torch.LongTensor(masks).view(-1, 1).cuda())  # (n, 1)
    scores = F.sigmoid(scores).view(-1)  # (n, )
    return scores