import collections
import torch
import torch.nn as nn
import numpy as np

# clip gradient
def clip_gradient(optimizer, grad_clip):
  for group in optimizer.param_groups:
    for param in group['params']:
      if hasattr(param.grad, 'data'):
        param.grad.data.clamp_(-grad_clip, grad_clip)

# reset learning rate
def set_lr(optimizer, lr):
  for group in optimizer.param_groups:
    group['lr'] = lr

# iou computation
def compute_iou(cand_mask, ref_mask=None):
  """
  Given (h, w) cand_mask, we wanna our ref_mask to be in the center of image,
  with [0.25h:0.75h, 0.25w:0.75w] occupied.
  """
  if ref_mask is None:
    h, w = cand_mask.shape[0], cand_mask.shape[1]
    ref_mask = np.zeros((h,w), np.int8)
    ref_mask[int(0.25*h):int(0.85*h), int(0.25*w):int(0.75*w)] = 1
  inter = (cand_mask > 0) & (ref_mask > 0)
  union = (cand_mask > 0) | (ref_mask > 0)
  iou = inter.sum() / (union.sum() + 1e-5)
  return iou

# compute iou given (ego_sem, fine_class, pos)
def compute_obj_iou(ego_sem, cls_to_rgb, fine_class):
  """
  Inputs:
  - ego_sem    : (h, w, 3) egocentric semantics map at [pos]
  - cls_to_rgb : fine_class -> rgb
  - fine_class : object's fine_class
  Output:
  - iou        : a value between 0 and 1
  """
  # global ref_mask (won't be changed)
  REF_MASK = np.zeros((224, 224), np.int8)
  REF_MASK[int(0.25*224):int(0.85*224), int(0.25*224):int(0.75*224)] = 1
  # compute iou given obj_cls
  c = np.array(cls_to_rgb[fine_class]).astype(np.uint8)    # (3, )
  obj_seg = np.all(ego_sem == c, axis=2).astype(np.uint8)  # (224, 224)
  iou = float(compute_iou(obj_seg, ref_mask=REF_MASK))
  return iou

def evaluate_precision_recall_fscore(predictions, ref_key='key_ixs', pred_key='sample_ixs', window_size=5):
  """
  Evaluates precision, recall, f1 for "storing" action.
  """
  assert len(predictions) > 0
  # eval acc for storing
  tp, fp, fn = 0, 0, 0
  for entry in predictions:
    for key_ix, sample_ix in zip(entry[ref_key], entry[pred_key]):
      if key_ix in range(sample_ix+1-window_size, sample_ix+1):
        tp += 1
      else:
        fn += 1   # at gd=1, predicted 0.
        fp += 1   # at gd=0, predicted 1.
  precision = tp / (tp + fp + 1e-5)
  recall = tp / (tp + fn + 1e-5)
  f1 = 2 * precision * recall / (precision + recall + 1e-5)
  return precision, recall, f1

def evaluate_distance_iou(predictions, ref_key='key_ixs', pred_key='sample_ixs', window_size=5):
  """
  Evaluates distance and iou for "storing" action.
  """
  assert len(predictions) > 0
  # iou = inter(w1, w2) / union(w1, w2)
  ious = []
  dists = []
  evals = 0
  for entry in predictions:
    for key_ix, sample_ix in zip(entry[ref_key], entry[pred_key]):
      # compute iou
      ref_window = range(max(0, key_ix+1-window_size), key_ix+1)
      pred_window = range(max(0, sample_ix+1-window_size), sample_ix+1)
      inter = set(ref_window).intersection(set(pred_window))
      union = set(ref_window).union(set(pred_window))
      iou = len(inter) / (len(union)+1e-5)
      ious.append(iou) 
      # compute distance
      dist = abs(key_ix - sample_ix)
      dists.append(dist)
      evals += 1
  return sum(ious)/evals, sum(dists)/evals

def evaluate_miss_rate(predictions, key_name='seq_probs'):
  """
  Evaluate how many localization is missed, i.e., seq_probs is always <= 0.5.
  """
  hit = 0
  evals = 0
  for entry in predictions:
    seq_probs = np.array(entry[key_name])
    hit += ((seq_probs >= 0.5).sum(1) > 0).sum()
    evals += seq_probs.shape[0]
  return 1-hit/evals

# def metric_precision_recall_fscore(predictions, ref_key='gd_key_ix', pred_key='pred_first_ix', window_size=5):
#   """
#   Evaluates precision, recall, f1 for "storing" action.
#   """
#   assert len(prediction) > 0
#   # eval acc for storing
#   tp, fp, fn = 0, 0, 0
#   for entry in predictions: 
#     if entry[ref_key] in range(entry[pred_key]+1-window_size, entry[pred_key]+1):
#       tp += 1
#     else:
#       fn += 1   # At gd=1, we predicted 0.
#       if entry[pred_key] != -1:
#         fp += 1 # if we predict 1 (not at gd=1), then it's a false alarm
#   precision = tp / (tp + fp + 1e-5)
#   recall = tp / (tp + fn + 1e-5)
#   f1 = 2 * precision * recall / (precision + recall + 1e-5)
#   return precision, recall, f1


# def metric_distance_iou(predictions, ref_key='gd_key_ix', pred_key='pred_first_ix', window_size=5):
#   """
#   Evaluates distance and iou for "storing" action.
#   """
#   assert len(predictions) > 0
#   # iou = inter(w1, w2) / union(w1, w2)
#   ious = []
#   dists = []
#   for entry in predictions:
#     # compute iou
#     ref_window = range(max(0, entry[ref_key]+1-window_size), entry[ref_key]+1)
#     pred_window = range(max(0, entry[pred_key]+1-window_size), entry[pred_key]+1)
#     inter = set(ref_window).intersection(set(pred_window))
#     union = set(ref_window).union(set(pred_window))
#     iou = len(inter) / (len(union)+1e-5)
#     ious.append(iou) 
#     # compute distance
#     ref_pos = entry[ref_key]
#     pred_pos = entry[pred_key] if entry[pred_key] >= 0 else entry['path_len']-1
#     dist = abs(ref_pos - pred_pos)
#     dists.append(dist)
#   return sum(ious)/len(predictions), sum(dists)/len(predictions), ious, dist
