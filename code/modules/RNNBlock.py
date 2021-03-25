
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNBlock(nn.Module):
    def __init__(self, cfg):
        super(RNNBlock, self).__init__()
        self.batch_size = cfg.batch_size
        self.model_in = cfg.model_in
        self.rnn_hid = cfg.rnn_hid
        self.n_layers = cfg.n_layers
        self.dropout = cfg.dropout
        self.bidirection = cfg.bidirection

        cfg.model_out = self.rnn_hid * self.n_layers * (int(self.bidirection) + 1)

        self.rnn = nn.LSTM(
            input_size=self.model_in,
            hidden_size=self.rnn_hid,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirection
        )
    
    def forward(self, x, x_len):
        b, seq_len, _ = x.size()
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        rnn_output, (hn, cn) = self.rnn(x)
        hn = hn.transpose(0, 1).reshape(b, -1)
        cn = cn.transpose(0, 1).reshape(b, -1)
        return rnn_output, hn, cn
