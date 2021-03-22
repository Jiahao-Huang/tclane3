import torch
import torch.nn as nn
from modules import EmbeddingBlock, TransformerBlock


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingBlock(cfg)
        self.transformer = TransformerBlock(cfg)
    
    def forward(self, X):
        batch_size = X.shape[0]
        transformer_output = self.transformer(X)
        return transformer_output.reshape(batch_size, -1)