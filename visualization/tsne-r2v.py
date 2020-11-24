import torch
import torch.nn as nn
import json
import os
import pickle
import numpy as np

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from models import ggnn_r2v_tsne

import json
imsitu = json.load(open("imsitu_space.json"))
nouns = imsitu["nouns"]

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

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=3)

args = parser.parse_args('')

batch_size = args.batch_size
n_worker = args.num_workers

dataset_folder = args.dataset_folder
imgset_folder = args.imgset_dir

train_set = json.load(open(dataset_folder + '/' + args.train_file))

encoder = imsitu_encoder.imsitu_encoder(train_set)

train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, encoder,'train', encoder.train_transform)

constructor = 'build_ggnn_baseline'
model = getattr(ggnn_r2v_tsne, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

test_set = json.load(open(dataset_folder + '/' + args.test_file))
test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

model.cuda()
utils.load_net('./trained_models/train_full_main_ggnn_r2v_w.model', [model])
model.eval()

features_list = []
labels_id = []
labels_name = []
with torch.no_grad():
    for i, (img_id, img, verb, labels) in enumerate(dev_loader):
        img = torch.autograd.Variable(img.cuda())
        verb = torch.autograd.Variable(verb.cuda())
        labels = torch.autograd.Variable(labels.cuda())
        role_predict, adj_r2v, features = model(img, verb)
        features = features.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()[0][0]
        for i in range(6):
            label = labels[i]
            if label == 2001:
                continue
            label_id = encoder.label_list[label]
            if label_id not in nouns.keys():
                continue
            label_name = nouns[label_id]
            labels_id.append(label)
            labels_name.append(label_name['gloss'][0])
            features_list.append(features[i].reshape(1,-1))
    features_list = np.concatenate(features_list,axis=0)
    labels_id = np.array(labels_id)
    labels_name = np.array(labels_name)
    np.save('tsne_features.npy',features_list)
    np.save('tsne_labels_id.npy',labels_id)
    np.save('tsne_labels_name.npy',labels_name)