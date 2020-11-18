import torch
import torch.nn as nn

class Top_Down_Baseline(nn.Module):
    def __init__(self, verb_model, role_model):
        super(Top_Down_Baseline, self).__init__()
        self.verb_model = verb_model
        self.role_model = role_model

    def forward(self, v_org, topk=5):

        verb_pred = self.verb_model(v_org, epoch_num=0, verb=None, is_train=False)

        role_pred_topk = None

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        verbs = sorted_idx[:,:topk]

        for k in range(0,topk):
            # # for graph model
            # role_pred = self.role_model(v_org, verbs[:,k])[0]
            # # for baseline model
            # role_pred = self.role_model(v_org, verbs[:,k])

            role_pred = self.role_model(v_org, verbs[:,k])
            if type(role_pred) == tuple:
                role_pred = role_pred[0]

            if k == 0:
                idx = torch.max(role_pred,-1)[1]
                role_pred_topk = idx
            else:
                idx = torch.max(role_pred,-1)[1]
                role_pred_topk = torch.cat((role_pred_topk.clone(), idx), 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return verbs, role_pred_topk

def build_verb_role_joint(verb_model, role_model):

    return Top_Down_Baseline(verb_model, role_model)