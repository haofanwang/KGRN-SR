'''
PyTorch implementation of GGNN based SR : https://arxiv.org/abs/1708.04320
GGNN implementation adapted from https://github.com/chingyaoc/ggnn.pytorch
'''
import math
import copy
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from lib.fc import FCNet

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

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn , torch.mean(scores,1)

class MultiHeadedAttention(nn.Module):
    '''
    Reused implementation from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    '''
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, mean_scores = attention(query, key, value, mask=mask,
                                              dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), torch.mean(self.attn, 1)

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features
        self.out_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        self.vgg_classifier = nn.Sequential(*features) # Replace the model classifier

        self.resize = nn.Sequential(
            nn.Linear(4096, 1024)
        )

    def forward(self,x):
        features = self.vgg_features(x)
        y = self.resize(self.vgg_classifier(features.view(-1, 512*7*7)))
        return y

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_node,  n_steps):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_node = n_node
        self.n_steps = n_steps

        #neighbour projection
        self.W_p = nn.Linear(state_dim, state_dim)
        #weights of update gate
        self.W_z = nn.Linear(state_dim, state_dim)
        self.U_z = nn.Linear(state_dim, state_dim)
        #weights of reset gate
        self.W_r = nn.Linear(state_dim, state_dim)
        self.U_r = nn.Linear(state_dim, state_dim)
        #weights of transform
        self.W_h = nn.Linear(state_dim, state_dim)
        self.U_h = nn.Linear(state_dim, state_dim)

    def forward(self, init_node, mask):

        hidden_state = init_node

        for t in range(self.n_steps):
            # calculating neighbour info
            neighbours = hidden_state.contiguous().view(mask.size(0), self.n_node, -1)
            neighbours = neighbours.expand(self.n_node, neighbours.size(0), neighbours.size(1), neighbours.size(2))
            neighbours = neighbours.transpose(0,1)

            neighbours = neighbours * mask.unsqueeze(-1)
            neighbours = self.W_p(neighbours)
            neighbours = torch.sum(neighbours, 2)
            neighbours = neighbours.contiguous().view(mask.size(0)*self.n_node, -1)

            #applying gating
            z_t = torch.sigmoid(self.W_z(neighbours) + self.U_z(hidden_state))
            r_t = torch.sigmoid(self.W_r(neighbours) + self.U_r(hidden_state))
            h_hat_t = torch.tanh(self.W_h(neighbours) + self.U_h(r_t * hidden_state))
            hidden_state = (1 - z_t) * hidden_state + z_t * h_hat_t

        return hidden_state

class GGNN_Baseline(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, ggnn, classifier, encoder, query_composer, G1, G2):
        super(GGNN_Baseline, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.ggnn = ggnn
        self.classifier = classifier
        self.encoder = encoder
        self.query_composer = query_composer
        self.G1 = G1
        self.G2 = G2

    def forward(self, v_org, gt_verb):

        img_features = self.convnet(v_org)

        v = img_features

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        # repeat single image for max role count a frame can have
        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1) # [batch_size*role_count, hidden_size]=[32*6, 1024]=[192,1024]
        img = img.view(batch_size, self.encoder.max_role_count, -1) # [batch_size,role_count, hidden_size]=[32,6,1024]
        # print("0",img.shape)
        verb_embd = self.verb_emb(gt_verb) # [batch_size, word_embedding_size]=[32,300]
        # print("1",verb_embd.shape)
        role_embd = self.role_emb(role_idx) # [batch_size, role_count, word_embedding_size]=[32,6,300]
        # print("2",role_embd.shape)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1) # [batch_size, role_count, word_embedding_size]=[32,6,300]
        # print("3",verb_embed_expand.shape)

        concat_query = torch.cat([verb_embed_expand, role_embd], -1) # [batch_size, role_count, word_embedding_size*2]=[32,6,600]
        # print("4",concat_query.shape)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd) # [batch_size*role_count, hidden_size]=[32*6,1024]=[192,1024]
        # print("5",q_emb.shape)

        mask = self.encoder.get_adj_matrix_noself(gt_verb) # [32,6,6]
        if torch.cuda.is_available():
            mask = mask.to(torch.device('cuda'))
        # print("6",mask.shape)
        
        # img: [32*6, 1024]
        # role_embd: [32,6,300]
        # verb_embed_expand: [32,6,300]
        # print("7",img.shape, role_embd.shape, verb_embed_expand.shape)

        cur_group = torch.cat([img,role_embd, verb_embed_expand],dim=-1) # [32,6,1024+300+300]

        # print("8",input2ggnn.shape)
        input2ggnn = cur_group.view(batch_size*self.encoder.max_role_count,-1) # [32*6,1024+300+300]

        out = self.ggnn(input2ggnn, mask) # [32*6,1024+300+300]
        # print("9",out.shape)

        # out_view = out.expand(3, out.size(0), out.size(1)) # [batch_size, 6, 1024+300+300]
        out_view = out.contiguous().view(batch_size, self.encoder.max_role_count, -1) # [batch_size, 6, 1024+300+300]

        feature1, adj_r2r = self.G1(out_view) # [batch_size,6,512], [batch_size,6,6]
        feature2, adj_r2v = self.G2(out_view) # [batch_size,6,512], [batch_size,6,6]

        feature1 = feature1.contiguous().view(batch_size*self.encoder.max_role_count,-1)
        feature2 = feature1.contiguous().view(batch_size*self.encoder.max_role_count,-1)
        # out_view = out_view.contiguous().view(batch_size*self.encoder.max_role_count,-1)

        out = torch.cat([feature1,feature2, out], dim=-1) # [batch_size*6, 1024+300+300+512+512]

        logits = self.classifier(out) # [batch_size*6, 2001]

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1) # [batch_size, 6, 2001]

        return role_label_pred, adj_r2r, adj_r2v

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels, adj_r2r, adj_r2v, gt_adj_r2r, gt_adj_r2v):

        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())

        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size*self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size*self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))

        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3
        
        gt_adj_r2r = gt_adj_r2r.detach()
        adj_r2r = adj_r2r.repeat(1,3,3)
        adj_r2r_loss = F.mse_loss(adj_r2r, gt_adj_r2r)

        gt_adj_r2v = gt_adj_r2v.detach()
        adj_r2v = adj_r2v.repeat(1,3,3)
        adj_r2v_loss = F.mse_loss(adj_r2v, gt_adj_r2v)

        return loss.mean() + adj_r2r_loss.mean() + adj_r2v_loss.mean()

def build_ggnn_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300

    covnet = tv.models.resnet50(pretrained=True)
    covnet.fc = nn.Linear(2048, 1024)

    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    ggnn = GGNN(state_dim = hidden_size+word_embedding_size*2, n_node=encoder.max_role_count, n_steps=4)

    G1 = Know_Rout_mod(input_features=hidden_size+word_embedding_size*2, output_features=512)
    G2 = Know_Rout_mod(input_features=hidden_size+word_embedding_size*2, output_features=512)

    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(hidden_size+word_embedding_size*2+512*2, num_ans_classes)
    )

    return GGNN_Baseline(covnet, role_emb, verb_emb, ggnn, classifier, encoder, query_composer, G1, G2)