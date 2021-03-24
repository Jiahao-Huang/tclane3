import torch
import torch.nn as nn
from modules import EmbeddingBlock, TransformerBlock


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.batch_size = cfg.batch_size
        self.transformer = TransformerBlock(cfg)
    
    def forward(self, X):
        transformer_output = self.transformer(X)
        return transformer_output.reshape(self.batch_size, -1)