# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pdb

import copy
import json
import time
import os, sys
import argparse
import numpy as np

class Metric():
    def __init__(self, info={}, metric_names=[], log_json=None):
        self.info = info
        self.metric_names = metric_names

        self.metrics = [[None,None,None] for _ in self.metric_names]  # mean, running mean, mean_t

        self.stats = []
        self.num_iters = 0

        self.log_json = log_json

    def update(self, values):
        assert isinstance(values, list)

        self.num_iters += 1
        current_stats = []

        for i in range(len(values)):
            if values[i] is None:
                current_stats.append(None)
                continue

            if isinstance(values[i], list) == False:
                values[i] = [values[i]]

            if self.metrics[i][0] == None:
                self.metrics[i][0] = np.mean(values[i])
                self.metrics[i][1] = np.mean(values[i])
                self.metrics[i][2] = np.mean(values[i])
            else:
                self.metrics[i][0] = (self.metrics[i][0] * (self.num_iters - 1) + np.mean(values[i])) / self.num_iters
                self.metrics[i][1] = 0.95 * self.metrics[i][1] + 0.05 * np.mean(values[i])
                self.metrics[i][2] = np.mean(values[i])

            self.metrics[i][0] = float(self.metrics[i][0])
            self.metrics[i][1] = float(self.metrics[i][1])
            self.metrics[i][2] = float(self.metrics[i][2])

            current_stats.append(self.metrics[i])

        self.stats.append(copy.deepcopy(current_stats))

    def get_stat_string(self, mode=1):

        stat_string = ''

        for k, v in self.info.items():
            stat_string += '[%s:%s]' % (k, v)

        stat_string += '[iters:%d]' % self.num_iters

        for i in range(len(self.metric_names)):
            if self.metrics[i][mode]:
                stat_string += '[%s:%.05f]' % (self.metric_names[i], self.metrics[i][mode])

        return stat_string

    def dump_log(self, predictions=None):

        if self.log_json == None:
            return False

        dict_to_save = {
            'metric_names': self.metric_names,
            'stats': self.stats,
            'predictions': predictions,
        }

        json.dump(dict_to_save, open(self.log_json, 'w'))

        return True

class VqaMetric(Metric):
    def __init__(self, info={}, metric_names=[], log_json=None):
        Metric.__init__(self, info, metric_names, log_json)

    def compute_ranks(self, scores, labels):
        accuracy = np.zeros(len(labels))
        ranks = np.full(len(labels), scores.shape[1])

        for i in range(scores.shape[0]):
            ranks[i] = scores[i].gt(scores[i][labels[i]]).sum() + 1
            if ranks[i] == 1:
                accuracy[i] = 1

        return accuracy, ranks

class NavMetric(Metric):
    def __init__(self, info={}, metric_names=[], log_json=None):
        Metric.__init__(self, info, metric_names, log_json)
