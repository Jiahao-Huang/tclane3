import torch
import torch.nn as nn
from modules import EmbeddingBlock, CNNBlock, ClassifierBlock

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.embedding = EmbeddingBlock(cfg)
        self.cnns = nn.ModuleList([CNNBlock(cfg) for _ in range(cfg.n_layers)])
        self.weight_embed = nn.Linear(cfg.model_in, cfg.model_in)
        self.classifier = ClassifierBlock(cfg)

    def forward(self, X):
        X1 = X['X1']
        X2 = X['X2']

        output1 = self.embedding(X1)
        output2 = self.embedding(X2)

        batch_size, seq_len, _ = output1.size()

        weight1 = self.weight_embed(output1.reshape(batch_size*seq_len, -1)).reshape(batch_size, seq_len, -1)
        weight2 = self.weight_embed(output2.reshape(batch_size*seq_len, -1)).view(batch_size, seq_len, -1)
        weighted_output1 = weight1 * output1
        weighted_output2 = weight2 * output2

        for layer in self.cnns:
            output1, weighted_output1_layer = layer(output1)
            output2, weighted_output2_layer = layer(output2)

            weighted_output1 += weighted_output1_layer
            weighted_output2 += weighted_output2_layer
        

        return self.classifier(torch.sum(weighted_output1, dim=1), torch.sum(weighted_output2, dim=1))