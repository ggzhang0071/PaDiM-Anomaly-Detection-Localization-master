import torch
import torch.nn as nn


def softmax(logits):
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=1).reshape(logits.shape[0], 1)

class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    # input is batch of softmax scores
    def forward(self, x_batch):
        l = 0
        for x in x_batch:
            l += torch.sum(x * torch.log(x))
        return l / x_batch.shape[0]
        #return torch.sum(x_batch*torch.log(x_batch), dim=1)

