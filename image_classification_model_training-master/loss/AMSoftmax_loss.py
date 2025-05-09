# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

__all__ = ['AMSoftmax']


class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30,
                 init_center=None):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        if init_center:
            model_path = 'checkpoints_merge/focalloss_64_no_ok_merge_class/v1/model_best.pth'
            model = get_model(model_path)
            fc_weights = dict(model.named_parameters())['fc.weight'].T
            self.W = torch.nn.Parameter(fc_weights.cuda())
            # self.W = torch.nn.Parameter(torch.from_numpy(np.load(init_center).T).cuda())
        else:
            self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes).cuda(), requires_grad=True)
        # self.W = torch.nn.Parameter(self.W.cuda())
        self.ce = nn.CrossEntropyLoss()
        # nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss, costh_m_s
