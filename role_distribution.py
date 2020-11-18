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

import matplotlib.pyplot as plt

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder

import logging

logging.basicConfig(filename ='role_distribution.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import argparse
parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
parser.add_argument("--gpuid", default=0, help="put GPU id > -1 in GPU mode", type=int)
parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
parser.add_argument('--imgset_dir', type=str, default='../context-aware-reasoning-for-sr-master/resized_256', help='Location of original images')
parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args('')

batch_size = args.batch_size

dataset_folder = args.dataset_folder
imgset_folder = args.imgset_dir

import json
imsitu = json.load(open("imsitu_space.json"))
nouns = imsitu["nouns"]

train_set = json.load(open(dataset_folder + '/' + args.train_file))
dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
test_set = json.load(open(dataset_folder + '/' + args.test_file))

encoder = imsitu_encoder.imsitu_encoder(train_set)

full_set = {**train_set, **dev_set, **test_set}

verb_num = len(encoder.verb_list)
label_num = len(encoder.label_list)

full_set = imsitu_loader.imsitu_loader(imgset_folder, full_set, encoder,'train', encoder.train_transform)
full_loader = torch.utils.data.DataLoader(full_set, batch_size=batch_size, shuffle=False, num_workers=8)

# key: role_id
# value: total appearance number
role_total = {}

for idx, (_, img, gt_verbs, gt_labels) in enumerate(full_loader):
    if idx == 492:
        break
    for i in range(batch_size):  
        gt_verb = gt_verbs[i] # [1]
        gt_label = gt_labels[i] # [3,6]
        gt_v = gt_verb
        role_set = encoder.get_role_ids(gt_v)
        gt_role_count = encoder.get_role_count(gt_v) # the number of role for this verb
        gt_role_list = encoder.verb2_role_dict[encoder.verb_list[gt_v]]
        for k in range(gt_role_count):
            temp = []
            # each img has 3 frames
            for r in range(3):
                gt_label_id = gt_label[r][k]
                # if the label is None, pass
                if gt_label_id == '':
                    continue
                if gt_label_id not in temp:
                    role_total[int(gt_label_id)] = role_total.get(int(gt_label_id), 0) + 1
                    temp.append(int(gt_label_id))

# sorted
sorted_role_total = sorted(role_total.items(), key=lambda x: x[1], reverse=True)
x = np.arange(len(role_total))
y = np.array([sorted_role_total[i][1] for i in range(len(sorted_role_total))])

label_frequency = []
label_names = []
for i in range(2001):
    label = sorted_role_total[i][0] # 345
    label_id = encoder.label_list[label] # n1324362
    if label_id == '' or label_id == 'UNK':
        continue
    label_name = nouns[label_id] # walking
    label_names.append(label_name['gloss'][0])
    label_frequency.append(sorted_role_total[i][1])

logging.info("label names")
logging.info(str(label_names))
logging.info("label frequency")
logging.info(str(label_frequency))

plt.figure(figsize=(12,8))
x = [i for i in range(2001)]
plt.xticks([], [])
plt.ylabel("Frequency")
plt.bar(x, y, color = "dodgerblue")
plt.savefig('role_distribution_all.png',dpi=500)
plt.close()