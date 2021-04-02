import torch
import torch.nn as nn
from modules import BertBlock, RNNBlock, ClassifierBlock

class Bert(nn.Module):
    def __init__(self, cfg):
        super(Bert, self).__init__()
        self.bert = BertBlock(cfg)
        self.rnn = RNNBlock(cfg)

        cfg.classifier_in = cfg.rnn_out
        self.classifier = ClassifierBlock(cfg)
    
    def forward(self, X):
        X1, X2, len1, len2 = X['X1'], X['X2'], X['len1'], X['len2']

        bert_output1 = self.bert(X1)
        bert_output2 = self.bert(X2)

        _, rnn_output1, _ = self.rnn(bert_output1, len1)
        _, rnn_output2, _ = self.rnn(bert_output2, len2)

        output = self.classifier(rnn_output1, rnn_output2)

        return output

