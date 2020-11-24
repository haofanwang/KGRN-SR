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

logging.basicConfig(filename ='role_pair_distribution.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

import pickle
prob_r2r_adj_matrix = pickle.load(open('../context-aware-reasoning-for-sr-master/prob_r2r_adj_matrix.pkl', 'rb'))

# key: role_id
# value: total appearance number
role_role = {}

for i in range(len(prob_r2r_adj_matrix)):
    for j in range(i+1,len(prob_r2r_adj_matrix)):
        role_role[(i,j)] = prob_r2r_adj_matrix[i][j]

# sorted
sorted_role_role = sorted(role_role.items(), key=lambda x: x[1], reverse=True)
x = np.arange(len(role_role))
y = np.array([sorted_role_role[i][1] for i in range(len(sorted_role_role))])

label_frequency = []
label_names = []
for i in range(2001):
    label = sorted_role_role[i][0] # (345,1100)
    label_id_1 = encoder.label_list[label[0]] # n1324362
    label_id_2 = encoder.label_list[label[1]] # n1324362
    if label_id_1 == '' or label_id_1 == 'UNK' or label_id_2 == '' or label_id_2 == 'UNK' :
        continue
    label_name_1 = nouns[label_id_1] # husband
    label_name_2 = nouns[label_id_2] # wife
    label_names.append(label_name_1['gloss'][0]+','+label_name_2['gloss'][0])
    label_frequency.append(round(float(sorted_role_role[i][1].numpy()),3))

logging.info("label names")
logging.info(str(label_names))
logging.info("label frequency")
logging.info(str(label_frequency))

# bars = plt.bar(range(len(label_frequency)), height=label_frequency, color='b', width=.4)
# xlocs, xlabs = plt.xticks()

# x=[i/2 for i in range(len(label_names))]
# xlocs=[i for i in x]
# xlabs=[i for i in x]

# i = 0
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x(), yval+0.02, label_names[i], rotation=90, fontsize=7)
#     i += 1

# plt.xticks([], [])
# plt.ylim(0,0.7)
# plt.ylabel("Frequency")
# plt.show()
# plt.savefig('role_pair_distribution.png',dpi=500)
# plt.close()
