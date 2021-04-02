import os
import torch
import torch.nn as nn
from transformers import BertModel


class BertBlock(nn.Module):
    def __init__(self, cfg):
        super(BertBlock, self).__init__()

        model_path = os.path.join(cfg.cwd, cfg.pretrained_model)
        cfg_path = os.path.join(cfg.cwd, cfg.model_config)
        self.bert = BertModel.from_pretrained(model_path, config=cfg_path)
        
    
    def forward(self, x):
        attention_mask = torch.gt(x, 0)
        output = self.bert(x, attention_mask=attention_mask).last_hidden_state
        return output
