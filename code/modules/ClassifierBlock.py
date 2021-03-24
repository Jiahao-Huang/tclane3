
import torch
import torch.nn as nn


class ClassifierBlock(nn.Module):
    def __init__(self, cfg):
        super(ClassifierBlock, self).__init__()
        self.use_dist = cfg.use_dist
        self.use_cos = cfg.use_cos
        self.cls_hid = cfg.cls_hid

        self.fc_in = cfg.model_out
        if self.use_dist:
            self.fc_in += 1
        if self.use_cos:
            self.fc_in += 1

        self.fc_classify = nn.Sequential(
            nn.Linear(self.fc_in, self.cls_hid), 
            nn.ReLU(),
            nn.Linear(self.cls_hid, 2)
        )
    
    def inner_product(self, y1, y2):
        y1 = y1.unsqueeze(-1)
        y2 = y2.unsqueeze(-1)
        res = torch.matmul(y1.transpose(1, 2), y2)
        return res.view(-1, 1)
    
    def forward(self, vec1, vec2):
        vec = vec1 + vec2
        if self.use_dist:
            delta = vec1 - vec2
            dist = torch.sqrt(self.inner_product(delta, delta))
            vec = torch.cat((vec, dist), dim=1)
        if self.use_cos:
            cos = self.inner_product(vec1, vec2) / torch.sqrt(self.inner_product(vec1, vec1)) / torch.sqrt(self.inner_product(vec2, vec2))
            vec = torch.cat((vec, cos), dim=1)
        output = self.fc_classify(vec)

        return output
