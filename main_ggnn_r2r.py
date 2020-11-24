import torch
import torch.nn as nn
import json
import os
import pickle
import numpy as np

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from models import ggnn_r2r

import logging

logging.basicConfig(filename ='./logs/main_ggnn_r2r.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, model_name, model_saving_name, eval_frequency=1000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []

    if gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]
        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model

    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

    prob_r2v_adj_matrix = pickle.load(open('prob_r2v_adj_matrix.pkl', 'rb'))
    prob_r2r_adj_matrix = pickle.load(open('prob_r2r_adj_matrix.pkl', 'rb'))

    prob_r2v_adj_matrix = nn.Parameter(prob_r2v_adj_matrix.float(),requires_grad=False)
    prob_r2r_adj_matrix = nn.Parameter(prob_r2r_adj_matrix.float(),requires_grad=False)

    for epoch in range(max_epoch):

        for i, (_, img, verb, labels) in enumerate(train_loader):
            total_steps += 1

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            batch_size = len(verb)
            
            temp_labels = labels.contiguous().view(batch_size, -1) # batch_size, 18

            gt_adj_r2v = torch.zeros(batch_size, temp_labels.size(1), temp_labels.size(1)).cuda() # batch_size, 18, 18
            gt_adj_r2r = torch.zeros(batch_size, temp_labels.size(1), temp_labels.size(1)).cuda() # batch_size, 18, 18

            new_tensor1 = torch.zeros(1, len(prob_r2v_adj_matrix))
            new_tensor2 = torch.zeros(1, temp_labels.size(1))
            for b in range(batch_size):
                new_list = []
                for i in range(len(temp_labels[b])):
                    if int(temp_labels[b][i]) == 2001:
                        new_list.append(new_tensor1)
                    else:
                        tmp = prob_r2v_adj_matrix[temp_labels[b][i], :].unsqueeze(dim=0)
                        new_list.append(tmp)
                temp = torch.cat(new_list, dim=0)
                new_list1 = []
                for i in range(len(temp_labels[b])):
                    if int(temp_labels[b][i]) == 2001:
                        new_list1.append(new_tensor2)
                    else:
                        tmp = temp.T[temp_labels[b][i], :].unsqueeze(dim=0)
                        new_list1.append(tmp)
                temp1 = torch.cat(new_list1, dim=0)
                gt_adj_r2v[b] = temp1.transpose(0,1)

            for b in range(batch_size):
                new_list = []
                for i in range(len(temp_labels[b])):
                    if int(temp_labels[b][i]) == 2001:
                        new_list.append(new_tensor1)
                    else:
                        tmp = prob_r2r_adj_matrix[temp_labels[b][i], :].unsqueeze(dim=0)
                        new_list.append(tmp)
                temp = torch.cat(new_list, dim=0)
                new_list1 = []
                for i in range(len(temp_labels[b])):
                    if int(temp_labels[b][i]) == 2001:
                        new_list1.append(new_tensor2)
                    else:
                        tmp = temp.T[temp_labels[b][i], :].unsqueeze(dim=0)
                        new_list1.append(tmp)
                temp1 = torch.cat(new_list1, dim=0)
                gt_adj_r2r[b] = temp1.transpose(0,1) 

            role_predict, adj_r2r = pmodel(img, verb)
            loss = model.calculate_loss(verb, role_predict, labels, adj_r2r, gt_adj_r2r)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            top1.add_point_noun(verb, role_predict, labels)
            top5.add_point_noun(verb, role_predict, labels)

            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results_nouns()
                top5_a = top5.get_average_results_nouns()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))
                logging.info("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results_nouns()
                top5_avg = top5.get_average_results_nouns()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                logging.info('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils.format_dict(top5_avg, '{:.2f}', '5-')))
                dev_score_list.append(avg_score)
                max_score = max(dev_score_list)

                if max_score == dev_score_list[-1]:
                    torch.save(model.state_dict(), model_dir + "/{}_{}.model".format( model_name, model_saving_name))
                    print ('New best model saved! {0}'.format(max_score))
                    logging.info('New best model saved! {0}'.format(max_score))

                print('current train loss', train_loss)
                logging.info('current train loss {}'.format(train_loss))
                train_loss = 0
                top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
                top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

            del role_predict, loss, img, verb, labels
        print('Epoch ', epoch, ' completed!')
        logging.info('Epoch {} completed!'.format(epoch))
        scheduler.step()

def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():

        for i, (img_id, img, verb, labels) in enumerate(dev_loader):

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            role_predict, adj_r2r = model(img, verb)

            if write_to_file:
                top1.add_point_noun_log(img_id, verb, role_predict, labels)
                top5.add_point_noun_log(img_id, verb, role_predict, labels)
            else:
                top1.add_point_noun(verb, role_predict, labels)
                top5.add_point_noun(verb, role_predict, labels)

            del role_predict, img, verb, labels

    return top1, top5, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
    parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
    parser.add_argument('--model_saving_name', default="main_ggnn_r2r", type=str, help='saving name of the outpul model')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model', type=str, default='ggnn_baseline')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=3)

    args = parser.parse_args()

    n_epoch = args.epochs
    batch_size = args.batch_size
    clip_norm = args.clip_norm
    n_worker = args.num_workers

    dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    train_set = json.load(open(dataset_folder + '/' + args.train_file))

    encoder = imsitu_encoder.imsitu_encoder(train_set)

    train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, encoder,'train', encoder.train_transform)

    constructor = 'build_%s' % args.model
    model = getattr(ggnn_r2r, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.manual_seed(args.seed)
    if args.gpuid >= 0:
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    if args.resume_training:
        print('Resume training from: {}'.format(args.resume_model))
        logging.info('Resume training from: {}'.format(args.resume_model))
        args.train_all = True
        if len(args.resume_model) == 0:
            raise Exception('[pretrained module] not specified')
        utils.load_net(args.resume_model, [model])
        optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
        model_name = 'resume_all'

    else:
        print('Training from the scratch.')
        logging.info('Training from the scratch.')
        model_name = 'train_full'
        utils.set_trainable(model, True)
        optimizer = torch.optim.Adamax([
            {'params': model.convnet.parameters(), 'lr': 5e-5},
            {'params': model.role_emb.parameters()},
            {'params': model.verb_emb.parameters()},
            {'params': model.ggnn.parameters()},
            {'params': model.query_composer.parameters()},
            {'params': model.G1.parameters()},
            {'params': model.classifier.parameters()}
        ], lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if args.evaluate:
        top1, top5, val_loss = eval(model, dev_loader, encoder, args.gpuid)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.2f}', '5-')))
        logging.info('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.2f}', '5-')))

    elif args.test:
        top1, top5, val_loss = eval(model, test_loader, encoder, args.gpuid)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                    utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                    utils.format_dict(top5_avg, '{:.2f}', '5-')))
        logging.info('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                    utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                    utils.format_dict(top5_avg, '{:.2f}', '5-')))

    else:
        print('Model training started!')
        logging.info('Model training started!')
        train(model, train_loader, dev_loader, optimizer, scheduler, n_epoch, args.output_dir, encoder, args.gpuid, clip_norm, model_name, args.model_saving_name,)

if __name__ == "__main__":
    main()