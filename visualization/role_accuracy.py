import os
import json
import copy
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from models import ggnn_baseline, ggnn_r2r_w, ggnn_r2v_w, top_down_baseline, top_down_query_context

logging.basicConfig(filename ='role_accuracy.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import json
imsitu = json.load(open("imsitu_space.json"))
nouns = imsitu["nouns"]

import argparse
parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
parser.add_argument("--gpuid", default=0, help="put GPU id > -1 in GPU mode", type=int)
parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
parser.add_argument('--imgset_dir', type=str, default='../context-aware-reasoning-for-sr-master/resized_256', help='Location of original images')
parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args('')

batch_size = args.batch_size

dataset_folder = args.dataset_folder
imgset_folder = args.imgset_dir

train_set = json.load(open(dataset_folder + '/' + args.train_file))
encoder = imsitu_encoder.imsitu_encoder(train_set)

dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

test_set = json.load(open(dataset_folder + '/' + args.test_file))
test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

full_set = {**json.load(open(dataset_folder + '/' + args.train_file)), **json.load(open(dataset_folder + '/' + args.dev_file)), **json.load(open(dataset_folder + '/' + args.test_file))}

verb_num = len(encoder.verb_list)
label_num = len(encoder.label_list)

full_set = imsitu_loader.imsitu_loader(imgset_folder, full_set, encoder,'train', encoder.train_transform)
full_loader = torch.utils.data.DataLoader(full_set, batch_size=batch_size, shuffle=False, num_workers=8)

# constructor = 'build_top_down_baseline'
# model = getattr(top_down_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
# utils.load_net('./trained_models/train_full_tda.model', [model])

# constructor = 'build_top_down_baseline'
# tda_model = getattr(top_down_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
# constructor = 'build_top_down_query_context'
# model = getattr(top_down_query_context, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder, tda_model)
# utils.load_net('./trained_models/train_full_caq.model', [model])

# constructor = 'build_ggnn_baseline'
# model = getattr(ggnn_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
# utils.load_net('./trained_models/train_full_main_ggnn_baseline_tda.model', [model])

# constructor = 'build_ggnn_baseline'
# model = getattr(ggnn_r2r_w, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
# utils.load_net('./trained_models/train_full_main_ggnn_r2r_w.model', [model])

constructor = 'build_ggnn_baseline'
baseline = getattr(ggnn_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
utils.load_net('./trained_models/train_full_main_ggnn_baseline_tda.model', [baseline])

constructor = 'build_ggnn_baseline'
model = getattr(ggnn_r2v_w, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
utils.load_net('./trained_models/train_full_main_ggnn_r2v_w.model', [model])

ngpus = 2
device_array = [i for i in range(0,ngpus)]
model = torch.nn.DataParallel(model, device_ids=device_array)
baseline = torch.nn.DataParallel(baseline, device_ids=device_array)

if args.gpuid >= 0:
    model.cuda()
    baseline.cuda()
    torch.backends.cudnn.benchmark = False

# key: role_id
# value: correct_time
role_correct = {}
role_correct_baseline = {}

# key: role_id
# value: total appearance number
role_total = {}

# key: role_id
# value: accuracy
role_accuracy = {}
role_accuracy_baseline = {}

model.eval()
baseline.eval()
with torch.no_grad():
    for idx, (_, img, gt_verbs, gt_labels) in enumerate(test_loader): 

        img = torch.autograd.Variable(img.cuda())
        gt_verbs = torch.autograd.Variable(gt_verbs.cuda())
        gt_labels = torch.autograd.Variable(gt_labels.cuda())  
        
        labels_predict = model(img, gt_verbs)
        labels_predict_baseline = baseline(img, gt_verbs)

        if type(labels_predict) == tuple:
            labels_predict = labels_predict[0]

        for i in range(batch_size):  
            gt_verb = gt_verbs[i] # [1]
            
            label_pred = labels_predict[i] # [6,2001]
            label_pred_baseline = labels_predict_baseline[i] # [6,2001]

            gt_label = gt_labels[i] # [3,6]
            gt_v = gt_verb
            role_set = encoder.get_role_ids(gt_v)
            gt_role_count = encoder.get_role_count(gt_v) # the number of role for this verb
            gt_role_list = encoder.verb2_role_dict[encoder.verb_list[gt_v]]
            for k in range(gt_role_count):
                # predicted label for the k-th role
                label_id = torch.max(label_pred[k],0)[1]
                label_id_baseline = torch.max(label_pred_baseline[k],0)[1]
                count = 0
                count_baseline = 0
                # each img has 3 frames
                for r in range(3):
                    gt_label_id = gt_label[r][k]
                    # if the label is None, pass
                    if gt_label_id == '':
                        continue
                    role_total[int(gt_label_id)] = role_total.get(int(gt_label_id), 0) + 1
                    if gt_label_id == label_id:
                        count += 1
                    if gt_label_id == label_id_baseline:
                        count_baseline += 1
                role_correct[int(label_id)] = role_correct.get(int(label_id), 0) + count
                role_correct_baseline[int(label_id_baseline)] = role_correct_baseline.get(int(label_id_baseline), 0) + count_baseline

# key: the role
for key in role_total.keys():
    role_accuracy[key] = float(role_correct.get(key,0)) / float(role_total[key])

# [(label_idx, frequency), (),...()]
sorted_role_total = sorted(role_total.items(), key=lambda x: x[1], reverse=True)

valid_role_id = []
for i in range(len(sorted_role_total)):
    valid_role_id.append(sorted_role_total[i][0])

# seperate role accuracy
valid_role_names = []
for i in range(len(sorted_role_total)):
    # if encoder.label_list[sorted_role_total[i][0]] in nouns.keys() and sorted_role_total[i][0] in role_correct.keys():
    #     valid_role_names.append(nouns[encoder.label_list[sorted_role_total[i][0]]]['gloss'][0])
    if encoder.label_list[sorted_role_total[i][0]] in nouns.keys():
        valid_role_names.append(nouns[encoder.label_list[sorted_role_total[i][0]]]['gloss'][0])

role_accuracy = []
role_accuracy_baseline = []
for i in range(len(sorted_role_total)):
    if encoder.label_list[sorted_role_total[i][0]] in nouns.keys():
        if sorted_role_total[i][0] in role_correct.keys() and sorted_role_total[i][0] in role_correct_baseline.keys():
            role_accuracy.append(round(role_correct[sorted_role_total[i][0]] / role_total[sorted_role_total[i][0]],2))
            role_accuracy_baseline.append(round(role_correct_baseline[sorted_role_total[i][0]] / role_total[sorted_role_total[i][0]],2))
        else:
            role_accuracy.append(0.0)
            role_accuracy_baseline.append(0.0)

print(role_accuracy)
print("#######")
print(role_accuracy_baseline)

# # high frequency
# print("top20 name:", valid_role_names[:20])
# print("top20 acc:", role_accuracy[:20], np.mean(role_accuracy[:20]))

# # low frequency
# low_frequency_index = [] 
# low_frequency_name = []
# low_frequency_accuracy = []
# low_frequency_accuracy_baseline = []

# for i in range(800,1950):
#     if len(low_frequency_accuracy) == 20:
#         break
#     if role_accuracy_baseline[i]<role_accuracy[i]:
#         low_frequency_name.append(valid_role_names[i])
#         low_frequency_accuracy.append(role_accuracy[i])
#         low_frequency_accuracy_baseline.append(role_accuracy_baseline[i])
#         low_frequency_index.append(i)

# print("low frequency index:",low_frequency_index)
# print("low frequency name:",low_frequency_name)
# print("low frequency accuracy:",low_frequency_accuracy)
# print("low frequency accuracy baseline:",low_frequency_accuracy_baseline)


# top10_correct = 0
# top10_total = 0
# for i in valid_role_id[:10]:
#     if i in role_correct.keys():
#         top10_correct += role_correct[i]
#     top10_total += role_total[i]
# print("top10 acc:", top10_correct / top10_total)

# top10_correct_rest = 0
# top10_total_rest = 0
# for i in valid_role_id[10:]:
#     if i in role_correct.keys():
#         top10_correct_rest += role_correct[i]
#     top10_total_rest += role_total[i]
# print("top10 rest acc:", top10_correct_rest / top10_total_rest)

# top20_correct = 0
# top20_total = 0
# for i in valid_role_id[:20]:
#     if i in role_correct.keys():
#         top20_correct += role_correct[i]
#     top20_total += role_total[i]
# print("top20 acc:", top20_correct / top20_total)

# top20_correct_rest = 0
# top20_total_rest = 0
# for i in valid_role_id[20:]:
#     if i in role_correct.keys():
#         top20_correct_rest += role_correct[i]
#     top20_total_rest += role_total[i]
# print("top20 rest acc:", top20_correct_rest / top20_total_rest)

# top50_correct = 0
# top50_total = 0
# for i in valid_role_id[:50]:
#     if i in role_correct.keys():
#         top50_correct += role_correct[i]
#     top50_total += role_total[i]
# print("top50 acc:", top50_correct / top50_total)

# top50_correct_rest = 0
# top50_total_rest = 0
# for i in valid_role_id[50:]:
#     if i in role_correct.keys():
#         top50_correct_rest += role_correct[i]
#     top50_total_rest += role_total[i]
# print("top50 rest acc:", top50_correct_rest / top50_total_rest)

# total_correct = 0
# total = 0
# for i in valid_role_id:
#     if i in role_correct.keys():
#         total_correct += role_correct[i]
#     total += role_total[i]
# print("total acc:", total_correct / total)