import os
import json
import time
import copy
import pickle
import logging
import numpy as np

import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from models import ggnn_baseline, ggnn_r2r_w, ggnn_r2v_w, top_down_baseline, top_down_query_context

start_time = time.time()

import json
imsitu = json.load(open("imsitu_space.json"))
nouns = imsitu["nouns"]

import argparse
parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
parser.add_argument("--gpuid", default=0, help="put GPU id > -1 in GPU mode", type=int)
parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
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

constructor = 'build_ggnn_baseline'
baseline = getattr(ggnn_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
utils.load_net('./trained_models/train_full_main_ggnn_baseline.model', [baseline])

constructor = 'build_ggnn_baseline'
model = getattr(ggnn_r2v_w, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
utils.load_net('./trained_models/train_full_main_ggnn_r2v.model', [model])

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

# TP(True Positive)
# key: role_id (label)
# value: time been predicted as role_id (pred)
role_TP = {}
role_TP_baseline = {}

# FP(False Positive)
# key: role_id (pred)
# value: other role_id (label)
role_FP = {}
role_FP_baseline = {}

# FN(False Negative)
# key: role_id (label)
# value: time beend predicted as other role_id (pred)
role_FN = {}
role_FN_baseline = {}

# key: role_id
# value: precision
role_precision = {}
role_precision_baseline = {}

# key: role_id
# value: recall
role_recall = {}
role_recall_baseline = {}

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

        # for each sample
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

                # each img has 3 frames
                for r in range(3):
                    gt_label_id = gt_label[r][k]

                    # if the label is None, pass
                    if gt_label_id == '':
                        continue
                    
                    if gt_label_id == label_id:
                        role_correct[int(gt_label_id)] = role_correct.get(int(gt_label_id), 0) + 1
                    else:
                        role_FP[int(label_id)] = role_FP.get(int(label_id), 0) + 1
                    if gt_label_id == label_id_baseline:
                        role_correct_baseline[int(gt_label_id)] = role_correct_baseline.get(int(gt_label_id), 0) + 1
                    else:
                        role_FP_baseline[int(label_id_baseline)] = role_FP_baseline.get(int(label_id_baseline), 0) + 1
                    role_total[int(gt_label_id)] = role_total.get(int(gt_label_id), 0) + 1

# TP
role_TP = role_correct
role_TP_baseline = role_correct_baseline
for key in role_total.keys():
    if key not in role_TP.keys():
        role_TP[key] = 0
    if key not in role_TP_baseline.keys():
        role_TP_baseline[key] = 0

# FP
role_FP = role_FP
role_FP_baseline = role_FP_baseline
for key in role_total.keys():
    if key not in role_FP.keys():
        role_FP[key] = 0
    if key not in role_FP_baseline.keys():
        role_FP_baseline[key] = 0

# FN
for key in role_total.keys():
    if key not in role_TP.keys():
        role_FN[key] = role_total[key]
    else:
        role_FN[key] = role_total[key] - role_TP[key]
    if key not in role_TP_baseline.keys():
        role_FN_baseline[key] = role_total[key]
    else:
        role_FN_baseline[key] = role_total[key] - role_TP_baseline[key]


# [(label_idx, frequency), (),...()], descending order
sorted_role_total = sorted(role_total.items(), key=lambda x: x[1], reverse=True)

# precision & recall (unsorted)
for key in role_total.keys():
    
    if float(role_TP.get(key,0)) == 0 or (float(role_TP[key])+float(role_FP[key])) == 0:
        role_precision[key] = 0
    else:
        role_precision[key] = float(role_TP.get(key,0)) / (float(role_TP[key])+float(role_FP[key]))
    
    if float(role_TP_baseline.get(key,0)) == 0 or (float(role_TP_baseline[key])+float(role_FP_baseline[key])) == 0:
        role_precision_baseline[key] = 0
    else:
        role_precision_baseline[key] = float(role_TP_baseline.get(key,0)) / (float(role_TP_baseline[key])+float(role_FP_baseline[key]))

    role_recall[key] = float(role_TP.get(key,0)) / (float(role_TP[key])+float(role_FN[key]))
    role_recall_baseline[key] = float(role_TP_baseline.get(key,0)) / (float(role_TP_baseline[key])+float(role_FN_baseline[key]))

print("Finished in {} secs".format(time.time()-start_time))


valid_role_id = []
for i in range(len(sorted_role_total)):
    valid_role_id.append(sorted_role_total[i][0])

role_precisions= []
role_recalls = []

role_precisions_baseline = []
role_recalls_baseline = []

for key in valid_role_id:
    
    role_precisions.append(role_precision[key])
    role_recalls.append(role_recall[key])

    role_precisions_baseline.append(role_precision_baseline[key])
    role_recalls_baseline.append(role_recall_baseline[key])

print("model, The precision of the 100 rarest nouns is {}".format(np.mean(role_precisions[-100:])))
print("model, The recall of the 100 rarest nouns is {}".format(np.mean(role_recalls[-100:])))

print("baseline, The precision of the 100 rarest nouns is {}".format(np.mean(role_precisions_baseline[-100:])))
print("baseline, The recall of the 100 rarest nouns is {}".format(np.mean(role_recalls_baseline[-100:])))

