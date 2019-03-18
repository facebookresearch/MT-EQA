import time
import h5py
import math
import argparse
import numpy as np
import os, sys, json

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- CNN -----------
class MultitaskCNN(nn.Module):
  def __init__(self, num_classes, pretrained, checkpoint_path):
    super(MultitaskCNN, self).__init__()

    self.num_classes = num_classes
    self.conv_block1 = nn.Sequential(
      nn.Conv2d(3, 8, 5), 
      nn.BatchNorm2d(8), 
      nn.ReLU(inplace=True), 
      nn.MaxPool2d(2, 2))
    self.conv_block2 = nn.Sequential(
      nn.Conv2d(8, 16, 5),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2))
    self.conv_block3 = nn.Sequential(
      nn.Conv2d(16, 32, 5),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2))
    self.conv_block4 = nn.Sequential(
      nn.Conv2d(32, 32, 5),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2))
    self.classifier = nn.Sequential(
      nn.Conv2d(32, 512, 5),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Dropout2d(),
      nn.Conv2d(512, 512, 1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.Dropout2d())
    # encoders for segmentation/depth/encoder
    self.encoder_seg = nn.Conv2d(512, self.num_classes, 1)
    self.encoder_depth = nn.Conv2d(512, 1, 1)
    self.encoder_ae = nn.Conv2d(512, 3, 1)
    # segmentation decoder
    self.score_pool2_seg = nn.Conv2d(16, self.num_classes, 1)
    self.score_pool3_seg = nn.Conv2d(32, self.num_classes, 1)
    # depth decoder 
    self.score_pool2_depth = nn.Conv2d(16, 1, 1)
    self.score_pool3_depth = nn.Conv2d(32, 1, 1)
    # encoder decoder 
    self.score_pool2_ae = nn.Conv2d(16, 3, 1)
    self.score_pool3_ae = nn.Conv2d(32, 3, 1)
    # load pretrained model parameters
    self.pretrained = pretrained
    if self.pretrained == True:
      print('Loading CNN weights from %s' % checkpoint_path)
      checkpoint = torch.load(checkpoint_path, map_location={'cuda:0': 'cpu'})
      self.load_state_dict(checkpoint['model_state'])
      # fix params
      for param in self.parameters():
        param.requries_grad = False
    else:
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * (m.out_channels + m.in_channels)
          m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
  
  def extract_feats(self, imgs, conv4_only=False):
    """
    Input:
    - imgs: (n, 3, 224, 224)
    Return 
    - conv4 features (n, 32, 10, 10)
    """
    conv1 = self.conv_block1(imgs)
    conv2 = self.conv_block2(conv1)  # (n, 16, 53, 53)
    conv3 = self.conv_block3(conv2)  # (n, 32, 24, 24)
    conv4 = self.conv_block4(conv3)  # (n, 32, 10, 10)
    if conv4_only:
      return conv4.view(-1, 3200)
    else:
      return conv2, conv3, conv4
  
  def compute_conv5(self, conv4):
    # return (n, 512, 6, 6) feats
    conv5 = self.classifier(conv4)   # (n, 512, 6, 6)
    return conv5
  
  def compute_conv5_segm(self, conv5):
    """
    Inputs:
    - conv5 (n, 512, 6, 6)
    Outputs:
    - segm  (n, 191, 6, 6)
    """
    # segmentation
    return self.encoder_seg(conv5)  # (n, 191, 6, 6)
  
  def compute_mid_segm(self, imgs):
    """
    Input:
    - imgs (n, 3, 224, 224)
    Output:
    - mid-size segmentation: (n, 191, 24, 24)
    """
    conv1 = self.conv_block1(imgs)
    conv2 = self.conv_block2(conv1)  # (n, 16, 53, 53)
    conv3 = self.conv_block3(conv2)  # (n, 32, 24, 24)
    conv4 = self.conv_block4(conv3)  # (n, 32, 10, 10)
    conv5 = self.classifier(conv4)
    encoder_output_seg = self.encoder_seg(conv5)  # (n, 191, 6, 6)
    score_pool3_seg = self.score_pool3_seg(conv3) # (n, 191, 24, 24)
    score_seg = F.upsample(encoder_output_seg, score_pool3_seg.size()[2:], mode='bilinear', align_corners=True)
    score_seg += score_pool3_seg  # (n, 191, 24, 24)
    return score_seg

  def compute_big_segm(self, imgs):
    """
    Input:
    - imgs (n, 3, 224, 224)
    Output:
    - big-size segmentatoin (n, 191, 53, 53)
    """
    conv1 = self.conv_block1(imgs)
    conv2 = self.conv_block2(conv1)  # (n, 16, 53, 53)
    conv3 = self.conv_block3(conv2)  # (n, 32, 24, 24)
    conv4 = self.conv_block4(conv3)  # (n, 32, 10, 10)
    conv5 = self.classifier(conv4)
    # segmentation
    encoder_output_seg = self.encoder_seg(conv5)   # (n, 191, 6, 6)
    score_pool3_seg = self.score_pool3_seg(conv3)  # (n, 191, 24, 24)
    score_seg = F.upsample(encoder_output_seg, score_pool3_seg.size()[2:], mode='bilinear', align_corners=True)
    score_seg += score_pool3_seg  # (n, 191, 24, 24)
    score_pool2_seg = self.score_pool2_seg(conv2)  # (n, 191, 53, 53)
    score_seg = F.upsample(score_seg, score_pool2_seg.size()[2:], mode='bilinear', align_corners=True)
    score_seg += score_pool2_seg  # (n, 191, 53, 53)
    return score_seg

  def compute_depth(self, imgs):
    """
    Inputs:
    - imgs (n, 3, 224, 224)
    Output:
    - out_depth (n, 1, 224, 224) after F.sigmoid() already
    """
    conv1 = self.conv_block1(imgs)
    conv2 = self.conv_block2(conv1)  # (n, 16, 53, 53)
    conv3 = self.conv_block3(conv2)  # (n, 32, 24, 24)
    conv4 = self.conv_block4(conv3)  # (n, 32, 10, 10)
    conv5 = self.classifier(conv4)
    # depth
    score_pool2_depth = self.score_pool2_depth(conv2)
    score_pool3_depth = self.score_pool3_depth(conv3)
    encoder_output_depth = self.encoder_depth(conv5)  # (n, 191, 6, 6)
    score_depth = F.upsample(encoder_output_depth, score_pool3_depth.size()[2:], mode='bilinear', align_corners=True)
    score_depth += score_pool3_depth
    score_depth = F.upsample(score_depth, score_pool2_depth.size()[2:], mode='bilinear', align_corners=True)
    score_depth += score_pool2_depth
    out_depth = F.sigmoid(F.upsample(score_depth, imgs.size()[2:], mode='bilinear', align_corners=True))  
    return out_depth

  def compute_segm_depth(self, imgs):
    """
    Inputs: 
    - imgs (n, 3, 224, 224)
    Outputs:
    - big-size segmentation (n, 191, 53, 53) score
    - out_depth (n, 224, 224) after F.sigmoid() already
    """
    conv1 = self.conv_block1(imgs)
    conv2 = self.conv_block2(conv1)  # (n, 16, 53, 53)
    conv3 = self.conv_block3(conv2)  # (n, 32, 24, 24)
    conv4 = self.conv_block4(conv3)  # (n, 32, 10, 10)
    conv5 = self.classifier(conv4)
    # segmentation
    encoder_output_seg = self.encoder_seg(conv5)   # (n, 191, 6, 6)
    score_pool3_seg = self.score_pool3_seg(conv3)  # (n, 191, 24, 24)
    score_seg = F.upsample(encoder_output_seg, score_pool3_seg.size()[2:], mode='bilinear', align_corners=True)
    score_seg += score_pool3_seg  # (n, 191, 24, 24)
    score_pool2_seg = self.score_pool2_seg(conv2)  # (n, 191, 53, 53)
    score_seg = F.upsample(score_seg, score_pool2_seg.size()[2:], mode='bilinear', align_corners=True)
    score_seg += score_pool2_seg  # (n, 191, 53, 53)
    # depth
    score_pool2_depth = self.score_pool2_depth(conv2)
    score_pool3_depth = self.score_pool3_depth(conv3)
    encoder_output_depth = self.encoder_depth(conv5)  # (n, 191, 6, 6)
    score_depth = F.upsample(encoder_output_depth, score_pool3_depth.size()[2:], mode='bilinear', align_corners=True)
    score_depth += score_pool3_depth
    score_depth = F.upsample(score_depth, score_pool2_depth.size()[2:], mode='bilinear', align_corners=True)
    score_depth += score_pool2_depth
    out_depth = F.sigmoid(F.upsample(score_depth, imgs.size()[2:], mode='bilinear', align_corners=True)) 
    # return
    return score_seg, out_depth.squeeze(1)
