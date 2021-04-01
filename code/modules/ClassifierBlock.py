
import torch
import torch.nn as nn


class ClassifierBlock(nn.Module):
    def __init__(self, cfg):
        super(ClassifierBlock, self).__init__()

        self.use_sum = cfg.use_sum
        self.use_dist = cfg.use_dist
        self.use_cos = cfg.use_cos

        fc_in = 0
        if cfg.use_sum:
            fc_in += cfg.classifier_in
        if cfg.use_dist:
            fc_in += 1
        if cfg.use_cos:
            fc_in += 1

        self.fc_classify = nn.Sequential(
            nn.Linear(fc_in, 2)
        )
    
    def inner_product(self, y1, y2):
        y1 = y1.unsqueeze(-1)
        y2 = y2.unsqueeze(-1)
        res = torch.matmul(y1.transpose(1, 2), y2)
        return res.view(-1, 1)
    
    def forward(self, vec1, vec2):
        cls_input = []
        if self.use_sum:
            vec_sum = torch.cat((vec, vec1 + vec2), dim=1)
            cls_input.append(vec_sum)
        if self.use_dist:
            delta = vec1 - vec2
            vec_dist = torch.sqrt(self.inner_product(delta, delta))
            cls_input.append(vec_dist)
        if self.use_cos:
            vec_cos = self.inner_product(vec1, vec2) / torch.sqrt(self.inner_product(vec1, vec1)) / torch.sqrt(self.inner_product(vec2, vec2))
            cls_input.append(vec_cos)
        
        vec = torch.cat(cls_input, dim=1)
        output = self.fc_classify(vec)

        return output
