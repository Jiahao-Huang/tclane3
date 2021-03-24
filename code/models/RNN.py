import torch
import torch.nn as nn
from modules import EmbeddingBlock, RNNBlock, ClassifierBlock

class RNN(nn.Module):
    def __init__(self, cfg):
        super(RNN, self).__init__()
        self.embedding = EmbeddingBlock(cfg)
        self.rnn = RNNBlock(cfg)
        self.classifier = ClassifierBlock(cfg)
    
    def forward(self, X):
        # unpack
        X1 = X['X1']
        X2 = X['X2']
        len1 = X['len1']
        len2 = X['len2']

        # embedding
        embedding_out1 = self.embedding(X1)
        embedding_out2 = self.embedding(X2)

        # rnn
        _, hn1, _ = self.rnn(embedding_out1, len1)
        _, hn2, _ = self.rnn(embedding_out2, len2)

        # classification
        output = self.classifier(hn1, hn2)


        return output
