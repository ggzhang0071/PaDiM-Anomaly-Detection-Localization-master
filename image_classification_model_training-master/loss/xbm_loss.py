import torch
from torch import nn
import torch.nn.functional as F
import time

__all__ = ['XBMLoss']


class XBM:
    def __init__(self):
        self.K = 8192  # cfg.XBM.SIZE
        self.feats = torch.zeros(self.K, 512).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


class XBMLoss(nn.Module):
    def __init__(self):
        super(XBMLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 2.0

    def forward(self, feats, targets, xbm_feats, xbm_targets):
        # targets [1, bs]
        # xbm_feats [1, bl], bl = buffer_length
        epsilon = 1e-5
        batch_size = feats.size(0)
        feats_norm = F.normalize(feats)
        # print(feats_norm[0])
        xbm_feats_norm = F.normalize(xbm_feats)
        sim_mat = torch.matmul(feats_norm, torch.t(xbm_feats_norm))

        # replace for loop with mask opreation
        # expand_targets = targets.repeat([batch_size, 1])
        # print('xbm_feats_shape', xbm_feats.shape)
        new_targets = targets.unsqueeze(-1)  # [1, bs] -> [bs, 1]
        xbm_targets = xbm_targets.repeat([batch_size, 1])  # [1, bl] -> [bs, bl]
        # xbm_targets = torch.reshape(xbm_targets, (-1, batch_size))  # [1, bl] -> [bl / bs, bs]
        pos_pair_ = torch.masked_select(sim_mat, xbm_targets == new_targets)
        pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
        neg_pair_ = torch.masked_select(sim_mat, xbm_targets != new_targets)
        if (pos_pair_.size()[0]==0) or (neg_pair_.size()[0] == 0):
            return torch.zeros([], requires_grad=True).cuda()
        neg_pair_ = neg_pair_ + self.margin
        min_pos = torch.min(pos_pair_)
        neg_pair = torch.masked_select(neg_pair_, neg_pair_> min_pos)
        # neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
        
        pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]
        if len(neg_pair) < 1 or len(pos_pair) < 1:
            return torch.zeros([], requires_grad=True).cuda()
        # weighting step
        pos_loss = (1.0 / self.scale_pos * torch.log(1 + 
                    torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))))
        # print('pos loss:', pos_loss)
        # print('pos pair mean:', torch.mean(pos_pair))
        neg_loss = (1.0 / self.scale_neg * torch.log(1 + 
                    torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))))
        # print('neg loss:', neg_loss)
        # print('neg pair mean:', torch.mean(neg_pair))
        loss = (pos_loss + neg_loss) / batch_size
        return loss
    
    
    # def forward(self, inputs_col, targets_col, inputs_row, target_row):
    #     batch_size = inputs_col.size(0)
    #     # print('targets_row:', target_row.shape)
    #     # sim_mat = torch.matmul(inputs_col, inputs_row.t())
    #     # inputs_col_norm = torch.norm(inputs_col, p=2, dim=1)
    #     # inputs_row_norm = torch.norm(inputs_row, p=2, dim=1)
    #     # print(inputs_col_norm.shape)
    #     # print(inputs_row_norm.shape)
    #     # end_norm = torch.dot(inputs_col_norm, inputs_row_norm.T)
    #     # sim_mat = torch.matmul(inputs_col, torch.t(inputs_row)) / end_norm
    #     inputs_col_norm = F.normalize(inputs_col)
    #     # print(inputs_col_norm[0])
    #     inputs_row_norm = F.normalize(inputs_row)
    #     sim_mat = torch.matmul(inputs_col_norm, torch.t(inputs_row_norm))

    #     epsilon = 1e-5
    #     loss = list()
    #     neg_count = 0

    #     for i in range(batch_size):
    #         pos_pair_ = torch.masked_select(sim_mat[i], target_row == targets_col[i])
    #         pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
    #         neg_pair_ = torch.masked_select(sim_mat[i], target_row != targets_col[i])
    #         if (pos_pair_.size()[0]==0) or (neg_pair_.size()[0] == 0):
    #             continue
    #         # sampling step
    #         neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
    #         pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]

    #         if len(neg_pair) < 1 or len(pos_pair) < 1:
    #             continue
    #         neg_count += len(neg_pair)

    #         # weighting step
    #         pos_loss = (
    #             1.0
    #             / self.scale_pos
    #             * torch.log(
    #                 1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
    #             )
    #         )
    #         neg_loss = (
    #             1.0
    #             / self.scale_neg
    #             * torch.log(
    #                 1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
    #             )
    #         )
    #         loss.append(pos_loss + neg_loss)

    #     if len(loss) == 0:
    #         return torch.zeros([], requires_grad=True).cuda()
    #     #log_info["neg_count"] = neg_count / batch_size
    #     #print(neg_count)
    #     loss = sum(loss) / batch_size
    #     return loss
        