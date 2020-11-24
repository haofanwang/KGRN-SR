import torch
import torch.nn as nn
import json
import os
import copy
import pickle
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder

def cout_w(prob, num, dim=1):
    prob_weight = prob[:, :num]
    sum_value = np.sum(prob_weight, keepdims=True, axis=dim) + 0.1
    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    return prob_weight

def cp_kl(a, b):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 1
    sum_ = a * np.log(a / b)
    all_value = [x for x in sum_ if str(x) != 'nan' and str(x) != 'inf']
    kl = np.sum(all_value)
    return kl

def compute_js(attr_prob):
    cls_num = attr_prob.shape[0]
    similarity = np.zeros((cls_num, cls_num))
    similarity[0, 1:] = 1
    similarity[1:, 0] = 1
    for i in range(1, cls_num):
        for j in range(1, cls_num):
            if i == j:
                similarity[i,j] = 0
            else:
                similarity[i,j] = 0.5 * (cp_kl(attr_prob[i, :], 0.5*(attr_prob[i, :] + attr_prob[j,:]))
                                         + cp_kl(attr_prob[j, :], 0.5*(attr_prob[i, :] + attr_prob[j, :])))
    return similarity

import argparse
parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
parser.add_argument("--gpuid", default=0, help="put GPU id > -1 in GPU mode", type=int)
parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args('')

batch_size = args.batch_size

dataset_folder = args.dataset_folder
imgset_folder = args.imgset_dir

train_set = json.load(open(dataset_folder + '/' + args.train_file))
dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
test_set = json.load(open(dataset_folder + '/' + args.test_file))

encoder = imsitu_encoder.imsitu_encoder(train_set)

full_set = {**train_set, **dev_set, **test_set}

verb_num = len(encoder.verb_list)
label_num = len(encoder.label_list)

r2r_adj_matrix = torch.zeros(label_num, label_num) # 2001,2001
v2r_adj_matrix = torch.zeros(verb_num, label_num) # 504,2001

for img_id in test_set:
    img = full_set[img_id]
    verb = img['verb']
    verb_idx = encoder.verb_list.index(verb)
    for frame in img['frames']:
        roles = frame.keys()
        roles_list = list(roles)
        for role in roles_list:
            label_idx = encoder.label_list.index(frame[role])
            v2r_adj_matrix[verb_idx][label_idx] += 1
        for i in range(len(roles)):
            if frame[roles_list[i]] == '':
                continue
            label_idx_i = encoder.label_list.index(frame[roles_list[i]])
            for j in range(len(roles)):
                if frame[roles_list[j]] == '':
                    continue
                label_idx_j = encoder.label_list.index(frame[roles_list[j]])
                r2r_adj_matrix[label_idx_i][label_idx_j] += 1 

# avoid nan
row_sum = r2r_adj_matrix.sum(1)
for i in range(label_num):
    r2r_adj_matrix[i, i] = row_sum[i] + 1.

# role to role probability matrix
prob_r2r_adj_matrix = torch.zeros(label_num, label_num)
for i in range(label_num):
    for j in range(label_num):
        prob_r2r_adj_matrix[i][j] = r2r_adj_matrix[i][j] / (np.sqrt(r2r_adj_matrix[i][i]) * np.sqrt(r2r_adj_matrix[j][j]))
pickle.dump(prob_r2r_adj_matrix, open('prob_r2r_adj_matrix.pkl', 'wb'))

# role to verb probability matrix
v2r_adj_matrix = v2r_adj_matrix.numpy()
prob_v2r_adj_matrix = cout_w(v2r_adj_matrix.T, num=len(v2r_adj_matrix.T))
prob_v2r_adj_matrix = compute_js(prob_v2r_adj_matrix)
prob_v2r_adj_matrix = 1 - prob_v2r_adj_matrix
prob_v2r_adj_matrix = torch.from_numpy(prob_v2r_adj_matrix)
pickle.dump(prob_v2r_adj_matrix, open('prob_r2v_adj_matrix.pkl', 'wb'))