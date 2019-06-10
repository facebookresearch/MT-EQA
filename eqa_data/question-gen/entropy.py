# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generates distribution of answers across all envs for a given question
"""

import csv
import argparse
import operator
import os, sys, json
import os.path as osp
from tqdm import tqdm
import numpy as np
import math

def computeEntropy(p_distr):
  n = len(p_distr)
  ent = 0.0
  if n == 1: return ent

  for p in p_distr:
    if p != 0.0:
      ent += -(p * math.log(p, n))
  return ent

def computeEntropyForQuestion(ans_freqs):
  return computeEntropy([freq for (ans, freq) in ans_freqs])


# takes a list of answers (across all envs) for a given question string
# and computes the (1) distribution and (2) normalized entropy for the dist
def computeAnswerDistributionForSingleQuestion(answers):
  answer_freq = dict()
  for ans in answers:
    if ans not in answer_freq: answer_freq[ans] = 1
    else: answer_freq[ans] = answer_freq[ans] + 1

  total_ans = sum(answer_freq.values())
  answer_freq = [ (ans, (cnt*1.)/total_ans) for (ans, cnt) in answer_freq.items() ]
  sorted_answer_freq = sorted(answer_freq, key=operator.itemgetter(1))
  sorted_answer_freq.reverse()
  return sorted_answer_freq, total_ans, computeEntropyForQuestion(sorted_answer_freq)

# call the computeAnswerDistributionForSingleQuestion() function for every qn
def computeAnswerDistributionForAllQuestions(all_answers):
  answer_distribution_all_qns = dict()
  qn_counts, qn_entropy = dict(), dict()

  for (question, answers) in all_answers.items():
    ans_dist, total_ans, ent = computeAnswerDistributionForSingleQuestion(answers)
    answer_distribution_all_qns[question] = ans_dist
    qn_counts[question] = total_ans
    qn_entropy[question] = ent

  return answer_distribution_all_qns, qn_counts, qn_entropy

# update dict storing all answers for a given question string
# with elem which is a (q, a) pair
def updateDict(qa_pair, all_answers):
  q, a = qa_pair[0] , qa_pair[1]
  if q not in all_answers: all_answers[q] = [a]
  else:
    ans_list = all_answers[q]
    ans_list.append(a)
    all_answers[q] = ans_list

"""
Structure of the JSON object (example JSON obj)
{
  "ques": "where is the refrigerator located"
  "ent": 0.3634764
  "count": "35"
  "answers": ["living room", "dining room", "kitchen"]
  "answer_freq": ["10", "15", "20"]
}
"""
def getObjForQuestion(question, answer_dist, count, entropy):
  answers = [ ans for (ans, freq) in answer_dist ]
  answer_freq = [ freq for (ans, freq) in answer_dist ]

  json_obj = dict()
  json_obj = {
    'ques': question,
    'count': count,
    'ent': entropy,
    'answers': answers,
    'answer_freq': answer_freq
  }
  return json_obj

def getJsonForAllQuestions(answer_distribution_all_qns, counts, entropy):
  question_stats_data = []
  for question in answer_distribution_all_qns.keys():
    ans_dist = answer_distribution_all_qns[question]
    json_obj = getObjForQuestion(
      question,
      ans_dist,
      counts[question],
      entropy[question]
    )
    question_stats_data.append(json_obj)
  return question_stats_data

def writeToJson(question_stats_data, file_name):
  with open(file_name, "w") as f:
    json.dump(question_stats_data, f)

def printInfoForAllQuestions(answer_distribution_all_qns, counts, entropy):
  print ("Distribution of answers across all questions:")
  for question in answer_distribution_all_qns.keys():
    ans_dist = answer_distribution_all_qns[question]
    print ("[%d][%f] %s" % (counts[question], entropy[question], question))
    for (ans, freq) in ans_dist:
      print ("\t%s : %f" % (ans, freq))

def collapseType(q_type):
  collapse = {
    'exist_positive': 'exist',
    'exist_negative': 'exist',
    'exist_logical_positive':'exist_logic',
    'exist_logical_negative_1':'exist_logic',
    'exist_logical_negative_2':'exist_logic',
    'exist_logical_or_negative':'exist_logic',
    'exist_logical_or_positive_1': 'exist_logic',
    'exist_logical_or_positive_2': 'exist_logic',
    'dist_compare_positive': 'dist_compare',
    'dist_compare_negative': 'dist_compare',
    'room_size_compare_positive': 'room_size_compare',
    'room_size_compare_negative': 'room_size_compare',
    'room_dist_compare_positive': 'room_dist_compare',
    'room_dist_compare_negative': 'room_dist_compare',
    'color_compare_same_room_positive': 'color_compare',
    'color_compare_same_room_negative': 'color_compare',
    'color_compare_cross_room_positive': 'color_compare',
    'color_compare_cross_room_negative': 'color_compare',
    'size_compare_same_room_positive': 'size_compare',
    'size_compare_same_room_negative': 'size_compare',
    'size_compare_cross_room_positive': 'size_compare',
    'size_compare_cross_room_negative': 'size_compare',
    }
  if q_type in collapse: q_type = collapse[q_type]
  return q_type

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--cacheDir', default='../cache/question-gen-outputs', help='directory for saving generated questions')
  parser.add_argument('--inputJson', default='questions_from_engine_v2.json', help='generated question json file')
  parser.add_argument('--qnStatsSubDir', default='entropy_stats', help='storing question statitics')
  args = parser.parse_args()

  # make output_dir
  output_dir = osp.join(args.cacheDir, args.qnStatsSubDir)
  if not osp.isdir(output_dir):
    os.makedirs(output_dir)

  # Load pre-generated json containing question/answers
  question_dataset = json.load(open(osp.join(args.cacheDir, args.inputJson), "r"))
  templates = set([ collapseType(q['type']) for q in question_dataset ])
  print ("%d questions loaded from JSON..." % len(question_dataset))

  # collect qa pairs
  master_answer_repo = dict()
  for template in templates: master_answer_repo[template] = dict()
  print ("Collating all questions and answers ...")

  for i in range(len(question_dataset)):
    q = question_dataset[i]
    q_type = collapseType(q['type'])
    updateDict((q['question'], q['answer']), master_answer_repo[q_type])

  for template in master_answer_repo.keys():
    print ("template: %s, # unique questions = %d"
      % (template, len(master_answer_repo[template])))

    answer_distribution_all_qns, qn_counts, qn_entropy = \
      computeAnswerDistributionForAllQuestions(master_answer_repo[template])
    
    question_stats_data = getJsonForAllQuestions(
      answer_distribution_all_qns,
      qn_counts,
      qn_entropy
    )

    output_file_path = osp.join(output_dir, 'question_stats_' + template + ".json")
    writeToJson(question_stats_data, output_file_path)

    # printInfoForAllQuestions(
    #   answer_distribution_all_qns,
    #   qn_counts,
    #   qn_entropy
    # )

