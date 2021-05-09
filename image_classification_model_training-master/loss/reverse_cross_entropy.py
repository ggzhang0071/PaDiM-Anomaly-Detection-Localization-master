import torch
import torch.nn as nn

class rev_cross_entropy(nn.Module):
    def __init__(self, class_num):
        super(rev_cross_entropy, self).__init__()
        self.class_num = class_num
    
    def forward(self, y_pred, y):
        batch_size = y.shape[0]
        ry = torch.ones(batch_size, self.class_num).cuda()
        for row_id, idx in enumerate(y):
            ry[row_id, idx] = 0
        ry /= (self.class_num - 1)
        val = 0
        for ry_s, log_y_pred_s in zip(ry, torch.log(y_pred)):
            val -= torch.matmul(ry_s, log_y_pred_s)
        return val / batch_size


if __name__ == '__main__':
    rce = rev_cross_entropy(9)
    y_pred = torch.rand(16, 9)
    y = torch.randint(0, 9, (16, 1))
    print(rce(y_pred, y))


