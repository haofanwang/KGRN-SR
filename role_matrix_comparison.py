import torch
import torch.nn.init
import torch.nn as nn
from torch.nn import functional as F

import os
import json
import copy
import shutil
import logging
import numpy as np

import torchvision.models as models
from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from models import ggnn_r2r_w, ggnn_r2v_w

import argparse
parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
parser.add_argument('--test', action='store_true', help='Only use the testing mode')
parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
parser.add_argument('--imgset_dir', type=str, default='../context-aware-reasoning-for-sr-master/resized_256', help='Location of original images')
parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--model', type=str, default='ggnn_baseline')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--clip_norm', type=float, default=0.25)
parser.add_argument('--num_workers', type=int, default=3)

args = parser.parse_args('')

n_epoch = args.epochs
batch_size = args.batch_size
clip_norm = args.clip_norm
n_worker = args.num_workers

dataset_folder = args.dataset_folder
imgset_folder = args.imgset_dir

train_set = json.load(open(dataset_folder + '/' + args.train_file))

encoder = imsitu_encoder.imsitu_encoder(train_set)

train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, encoder,'train', encoder.train_transform)

constructor = "build_ggnn_baseline"

# R2V
model_r2v = getattr(ggnn_r2v_w, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
utils.load_net('./trained_models/train_full_main_ggnn_r2v_w.model', [model_r2v])

# R2R
model_r2r = getattr(ggnn_r2r_w, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)
utils.load_net('./trained_models/train_full_main_ggnn_r2r_w.model', [model_r2r])

ngpus = 2
device_array = [i for i in range(0,ngpus)]
model_r2v = torch.nn.DataParallel(model_r2v, device_ids=device_array)
model_r2r = torch.nn.DataParallel(model_r2r, device_ids=device_array)

model_r2v.cuda()
model_r2r.cuda()
torch.backends.cudnn.benchmark = False

print("model loaded")

dev_set_raw = json.load(open(dataset_folder + '/' + args.dev_file))
dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set_raw, encoder, 'val', encoder.dev_transform)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=n_worker)

test_set_raw = json.load(open(dataset_folder + '/' + args.test_file))
test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set_raw, encoder, 'test', encoder.dev_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=n_worker)

model_r2v.eval()
model_r2r.eval()

import json
imsitu = json.load(open("imsitu_space.json"))
nouns = imsitu["nouns"]

flag1 = False
flag2 = False
count = 0
with torch.no_grad():
    for _, (img_id, img, verb, labels) in enumerate(test_loader):

        if flag1 and flag2 and count > 10:
            break
        
        img = torch.autograd.Variable(img.cuda())
        verb = torch.autograd.Variable(verb.cuda())
        labels = torch.autograd.Variable(labels.cuda())

        # R2V
        role_predict_r2v, r2v_matrix = model_r2v(img, verb)
        role_predict_r2v = torch.argmax(F.softmax(role_predict_r2v[0],1),1)
        role_predict_r2v_id = [encoder.label_list[i] for i in role_predict_r2v]
        role_predict_r2v_name = []
        for i in role_predict_r2v_id:
            if i == '' or i == 'UNK':
                role_predict_r2v_name.append(i)
            else:
                role_predict_r2v_name.append(nouns[i]['gloss'][0])
        
        # R2R
        role_predict_r2r, r2r_matrix = model_r2r(img, verb)
        role_predict_r2r = torch.argmax(F.softmax(role_predict_r2r[0],1),1)
        role_predict_r2r_id = [encoder.label_list[i] for i in role_predict_r2r]
        role_predict_r2r_name = []
        for i in role_predict_r2r_id:
            if i == '' or i == 'UNK':
                role_predict_r2r_name.append(i)
            else:
                role_predict_r2r_name.append(nouns[i]['gloss'][0])

        batch_size = len(verb)
        img_name = img_id[0]
        img_json_line = test_set_raw[img_name]
        verb_name = img_json_line['verb']
        img_frames = img_json_line['frames']
        role_names = list(img_frames[0].keys())

        gt_names = []
        for role_name in role_names:
            role_name_list = []
            for i in range(len(img_frames)):
                img_frame = img_frames[i]
                if img_frame[role_name] in nouns.keys():
                    role_name_list.append(nouns[img_frame[role_name]]['gloss'][0])
                else:
                    role_name_list.append('')
            gt_names.append(role_name_list)
        
        r2v_correct = 0
        r2r_correct = 0

        for i in range(len(role_names)):
            if role_predict_r2v_name[i] in gt_names[i]:
                r2v_correct += 1
            if role_predict_r2r_name[i] in gt_names[i]:
                r2r_correct += 1
        
        if r2v_correct == len(role_name):

            r2v_matrix = F.softmax(r2v_matrix[0],1)
            
            print("R2V")
            print('img name:{}'.format(img_name))
            print('verb name:{}'.format(verb_name))
            print('role name:{}'.format(str(role_names)))
            print('gt label:{}'.format(str(gt_names)))
            print('r2v label:{}'.format(str(role_predict_r2v_name)))
            print('total role:{}'.format(str(len(role_names))))
            print('r2v matrix:', r2v_matrix)
            shutil.copy('../context-aware-reasoning-for-sr-master/resized_256/'+img_name, './examples/'+img_name)
            flag1 = True
        
        if r2r_correct == len(role_name):
            r2r_matrix = F.softmax(r2r_matrix[0],1)

            print("R2R")
            print('img name:{}'.format(img_name))
            print('verb name:{}'.format(verb_name))
            print('role name:{}'.format(str(role_names)))
            print('gt label:{}'.format(str(gt_names)))
            print('r2r label:{}'.format(str(role_predict_r2r_name)))
            print('total role:{}'.format(str(len(role_names))))
            print('r2r matrix:', r2r_matrix)
            shutil.copy('../context-aware-reasoning-for-sr-master/resized_256/'+img_name, './examples/'+img_name)
            flag2=True
            count += 1