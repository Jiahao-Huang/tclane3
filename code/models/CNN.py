import torch
import torch.nn as nn

from modules import Embedding

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()

        self.embedding = Embedding(cfg)
    
    def forward(self, X):
        x = X['x']
        word_embedding = self.embedding(x)
        assert 0
        return 0
