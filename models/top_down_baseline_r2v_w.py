'''
This is the baseline model.
We directly use top-down VQA like mechanism for SR
Modified bottom-up top-down code from https://github.com/hengyuan-hu/bottom-up-attention-vqa and
added normalization from https://github.com/yuzcccc/vqa-mfb
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.attention import Attention
from lib.classifier import SimpleClassifier
from lib.fc import FCNet
import torchvision as tv

class A_compute(nn.Module):
    def __init__(self, input_features, nf=64, ratio=[4, 2, 1]):
        super(A_compute, self).__init__()
        self.num_features = nf
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        #        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), int(nf * ratio[2]), 1, stride=1)
        self.conv2d_4 = nn.Conv2d(int(nf * ratio[2]), 1, 1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cat_feature):
        W1 = cat_feature.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2)
        W_new = torch.abs(W1 - W2)
        W_new = torch.transpose(W_new, 1, 3)
        W_new = self.conv2d_1(W_new)
        W_new = self.relu(W_new)
        W_new = self.conv2d_2(W_new)
        W_new = self.relu(W_new)
        W_new = self.conv2d_3(W_new)
        W_new = self.relu(W_new)
        W_new = self.conv2d_4(W_new)
        W_new = W_new.contiguous()
        W_new = W_new.squeeze(1)
        Adj_M = W_new
        return Adj_M

class Know_Rout_mod(nn.Module):
    def __init__(self, input_features, output_features):
        super(Know_Rout_mod, self).__init__()
        self.input_features = input_features
        self.lay_1_compute_A = A_compute(input_features)
        self.transferW = nn.Linear(input_features, output_features)

    def forward(self, cat_feature):
        cat_feature_stop = torch.autograd.Variable(cat_feature.data)
        Adj_M1 = self.lay_1_compute_A(cat_feature_stop)
        # batch matrix-matrix product
        W_M1 = F.softmax(Adj_M1, 2)

        # batch matrix-matrix product
        cat_feature = torch.bmm(W_M1, cat_feature)
        cat_feature = self.transferW(cat_feature)

        return cat_feature, Adj_M1

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def forward(self,x):
        features = self.vgg_features(x)
        return features

def split_resnet50(model):
    return nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)

class Top_Down_Baseline(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, classifier, encoder, Dropout_C, G2):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.Dropout_C = Dropout_C
        self.G2 = G2

    def forward(self, v_org, gt_verb):
        '''
        :param v_org: original image
        :param gt_verb: ground truth verb id
        :return: predicted role label logits
        '''

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        #query for image reasoning
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)

        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        #normalization to avoid model convergence to unsatisfactory local minima
        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, 1)
        # sum pooling can be more useful if there are multiple heads like original MFB.
        # we kept out head count to 1 for final implementation, but experimented with multiple without considerable improvement.
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)
        mfb_out = torch.squeeze(mfb_iq_sumpool)
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2

        out_view = out.contiguous().view(batch_size, self.encoder.max_role_count, -1)

        feature2, adj_r2v = self.G2(out_view)
        feature2 = feature2.contiguous().view(batch_size*self.encoder.max_role_count,-1)
        out = torch.cat([feature2, out], dim=-1)

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return role_label_pred, adj_r2v

    def forward_hiddenrep(self, v_org, gt_verb):

        '''
        :param v_org: original image
        :param gt_verb: ground truth verb id
        :return: hidden representation which is the input to the classifier
        '''

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)

        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, 1)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2

        return out

    def forward_agentplace_noverb(self, v_org, pred_verb):

        max_role_count = 2

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_agent_place_ids_batch(batch_size)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(max_role_count, img.size(0), img.size(1), img.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * max_role_count, -1, v.size(2))

        #verb_embd = torch.sum(self.verb_emb.weight, 0)
        #verb_embd = verb_embd.expand(batch_size, verb_embd.size(-1))
        #verb_embd = torch.zeros(batch_size, 300).cuda()
        verb_embd = self.verb_emb(pred_verb)

        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)

        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* max_role_count, 1, -1, 1)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), max_role_count, -1)
        role_label_rep = v_repr.contiguous().view(v.size(0), max_role_count, -1)

        return role_label_pred, role_label_rep

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels, adj_r2v, gt_adj_r2v):

        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())

        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))

        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3

        gt_adj_r2v = gt_adj_r2v.detach()
        adj_r2v = adj_r2v.repeat(1,3,3)
        adj_r2v_loss = F.mse_loss(adj_r2v, gt_adj_r2v)

        return loss.mean() + adj_r2v_loss.mean()

def build_top_down_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 2048

    # covnet = vgg16_modified()
    
    covnet = split_resnet50(tv.models.resnet50(pretrained=True))
    
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])

    G2 = Know_Rout_mod(input_features=hidden_size+word_embedding_size*2, output_features=512)

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    Dropout_C = nn.Dropout(0.1)

    return Top_Down_Baseline(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                                                           v_net, classifier, encoder, Dropout_C, G2)


