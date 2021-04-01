import torch
import torch.nn as nn
from modules import EmbeddingBlock, TransformerBlock, RNNBlock, ClassifierBlock


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingBlock(cfg)

        cfg.transformer_in = cfg.embedding_dim
        self.transformer = TransformerBlock(cfg)

        cfg.rnn_in = cfg.transformer_out
        self.rnn = RNNBlock(cfg)

        cfg.classifier_in = cfg.rnn_out
        self.classifier = ClassifierBlock(cfg)
    
    def forward(self, X):
        X1, X2, len1, len2 = X['X1'], X['X2'], X['len1'], X['len2']
        mask1 = X1.data.eq(0)
        mask2 = X2.data.eq(0)

        embedding_output1 = self.embedding(X1)
        embedding_output2 = self.embedding(X2)

        transformer_output1 = self.transformer(embedding_output1.transpose(0, 1), mask1).transpose(0, 1)
        transformer_output2 = self.transformer(embedding_output2.transpose(0, 1), mask2).transpose(0, 1)

        _, rnn_output1, _ = self.rnn(transformer_output1, len1)
        _, rnn_output2, _ = self.rnn(transformer_output2, len2)
        output = self.classifier(rnn_output1, rnn_output2)
        return output