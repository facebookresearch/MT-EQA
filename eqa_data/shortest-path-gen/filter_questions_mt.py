"""
In "filter_questions_bak.py" we called entropy functions to filter those questions with
lower entropy (<0.5), however this is not big enough.
For example, 9 vs 61 would result in an entropy (0.55) that passed the threshold, though
it's quite peakly.
However, if we set too high threshold, we would get quite few questions in the end.

So my solution is:
1) compute entropy for each question and maintain those with big entropy (>0.9).
2) randomly kick out those (q, a) pairs with some probability, to make q's answers balanced.
This would leave me more questions, cheers!
"""
import json
import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm 
import argparse

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../question-gen'))
from entropy import *
from entropy_based_filtering import *

# Here are some bad objects found by Licheng.
# bad_object_list = ['partition', 'tv_stand', 'shower']
bad_object_list = []
Entropy_Threshold = 0.90  # all mt questions are just yes/no, we need to have a more strict entropy threshold
Count_Threshold = 2

def compute_questions_entropy(questions):
  templates = set([collapseType(qn['type']) for qn in questions])
  master_answer_repo = dict()
  for template in templates:
    master_answer_repo[template] = dict()
  
  for i in range(len(questions)):
    qn = questions[i]
    q_type = collapseType(qn['type'])
    updateDict((qn['question'], qn['answer']), master_answer_repo[q_type])
  
  master_ent = {}
  for template in master_answer_repo.keys():
    print ("template: %s, # unique questions = %d" % (template, len(master_answer_repo[template])))

    answer_distribution_all_qns, qn_counts, qn_entropy = \
      computeAnswerDistributionForAllQuestions(master_answer_repo[template])

    question_stats_data = getJsonForAllQuestions(
      answer_distribution_all_qns,
      qn_counts,
      qn_entropy
    )

    master_ent[template] = {}
    for obj in question_stats_data:
      master_ent[template][obj['ques']] = (obj['ent'], obj['count'], obj['answers'], obj['answer_freq'])
  
  return master_ent

def entropy_filter(questions):

  print('Before filtering, we have %s questions.' % len(questions))

  # compute template -> question -> {ent, count, answers, answer_freq}
  master_ent = compute_questions_entropy(questions)
  
  # filter by entropy
  ent_filtered_questions = []
  for qn in questions:
    if qn['question'] in master_ent[collapseType(qn['type'])]:
      ent, count, answers, answer_freq = master_ent[collapseType(qn['type'])][qn['question']]
      if ent > Entropy_Threshold:
        ent_filtered_questions.append(qn)
      else:
        ans_to_freq = {ans: answer_freq[i] for i, ans in enumerate(answers)}
        ans = qn['answer']
        if ans_to_freq[ans] < 0.5:
          ent_filtered_questions.append(qn)
        else:
          # if p = (7/9 - 2/9) / (7/9), we throw away (q, yes) pair with this probability to make yes/no balanced.
          p = min(answer_freq) / max(answer_freq)
          if np.random.uniform(0, 1, 1) < p:
            ent_filtered_questions.append(qn)
  print('After entropy filtering, we have %s questions.' % len(ent_filtered_questions))

  # compute entropy again 
  master_ent = compute_questions_entropy(ent_filtered_questions)

  # filter by count and entropy again
  cnt_filtered_questions = []
  for qn in ent_filtered_questions:
    if qn['question'] in master_ent[collapseType(qn['type'])]:
      ent, count, answers, answer_freq = master_ent[collapseType(qn['type'])][qn['question']]
      if count >= Count_Threshold and ent >= Entropy_Threshold:
        qn['entropy'] = ent
        qn['count'] = count
        cnt_filtered_questions.append(qn)
  print('After count and entropy filtering, we have %s questions.' % len(cnt_filtered_questions)) 

  return cnt_filtered_questions

def run_stats(questions, qid_to_samples=None):
  # qtype --> number of questions
  Questions = {qn['id']: qn for qn in questions}
  qtype_to_cnt = {}
  for qn in questions:
    qtype = qn['type']
    qtype_to_cnt[qtype] = qtype_to_cnt.get(qtype, 0) + 1
  for qtype, cnt in qtype_to_cnt.items():
    print('[%s] has %s questions.' % (qtype, cnt))
  
  # split --> number of questions
  print('\n')
  split_to_cnt, split_to_hids = {}, {}
  for qn in questions:
    split_to_cnt[qn['split']] = split_to_cnt.get(qn['split'], 0) + 1
    if qn['split'] not in split_to_hids: split_to_hids[qn['split']] = []
    split_to_hids[qn['split']].append(qn['house'])
  total_cnt = 0
  for split, cnt in split_to_cnt.items():
    print('[%s] has %s questions for %s houses.' % (split, cnt, len(set(split_to_hids[split])) ))
    total_cnt += cnt
  print('In all there are %s questions left.' % total_cnt)

  # qtype --> number of samples per question
  print('\n')
  if qid_to_samples is not None:
    qtype_to_num_samples = {}
    for qid, samples in qid_to_samples.items():
      qtype = Questions[qid]['type']
      qtype_to_num_samples[qtype] = qtype_to_num_samples.get(qtype, 0) + len(samples)
    for qtype, num_samples in qtype_to_num_samples.items():
      print('[%s] has %.2f samples per question.' % (qtype, num_samples/qtype_to_cnt[qtype]))

def main(args):

  # set seed
  np.random.seed(args.seed)

  # splits
  splits = json.load(open(args.splits_json))
  hid_to_split = {hid: split for split, hids in splits.items() for hid in hids}

  # all questions
  assert osp.exists(args.input_question_json)
  input_questions = json.load(open(args.input_question_json))
  Questions = {qn['id']: qn for qn in input_questions}

  # shortest_path_dir saves house.tid.tid.json, contains list of [{positions, actions, ordered, best_ious}]
  shortest_path_dir = osp.join(args.shortest_path_dir, args.sample_mode)
  assert osp.isdir(shortest_path_dir), '%s not exits.' % shortest_path_dir
  qids_with_spaths = []
  for qn in input_questions:
    path_name = qn['path'] + '.json'
    path_file = osp.join(shortest_path_dir, path_name)
    if osp.exists(path_file):
      qids_with_spaths.append(qn['id'])

  # filter out those w/o sampled pats
  filtered_questions = [Questions[qid] for qid in qids_with_spaths]

  # filter out those bad objects
  idx = [] 
  for i, qn in enumerate(filtered_questions):
    if 'object' in qn['type']:
      for box in qn['bbox']:
        if box['name'] in bad_object_list:
          idx.append(i)
          break
  print('%s questions are with bad objects.' % len(idx))
  for i in idx[::-1]:
    del filtered_questions[i]

  # add split
  for qn in filtered_questions:
    assert qn['house'] in hid_to_split
    qn['split'] = hid_to_split[qn['house']]

  # filter again by entropy
  # we need this again just after the shortest-path filtering!
  filtered_questions = entropy_filter(filtered_questions)

  # # filter the training set whose count is <2
  # del_ixs = []
  # for i, qn in enumerate(filtered_questions):
  #   if qn['split'] == 'train' and qn['count'] < args.min_cnt:
  #     del_ixs.append(i)
  # for ix in del_ixs[::-1]:
  #   del filtered_questions[ix]

  # save
  output_question_json = osp.join(args.output_dir, 'questions_mt_paths_'+args.sample_mode+'_program.json')
  with open(output_question_json, 'w') as f:
    json.dump(filtered_questions, f)
  print('Filtered questions (with sampled paths) saved as %s.\n' % output_question_json)

  # stats
  qid_to_samples = {}
  for qn in filtered_questions:
    path_name = qn['path'] + '.json'
    path_file = osp.join(shortest_path_dir, path_name)
    qid_to_samples[qn['id']] = json.load(open(path_file))
  run_stats(filtered_questions, qid_to_samples)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', default=24)
  parser.add_argument('--min_cnt', default=4)
  parser.add_argument('--splits_json', default='data/eqa_v1/splits.json')
  parser.add_argument('--input_question_json', default='cache/question-gen-outputs/questions_pruned_mt_with_conn_program.json')
  parser.add_argument('--shortest_path_dir', default='cache/shortest-paths-mt', help='directory saving shortest paths for every question')
  parser.add_argument('--sample_mode', default='nearby_source_best_view', type=str, help='nearby(random)_source_best_coverage(iou)')
  parser.add_argument('--output_dir', default='cache/question-gen-outputs')
  args = parser.parse_args()

  main(args)
