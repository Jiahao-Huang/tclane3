
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNBlock(nn.Module):
    def __init__(self, cfg):
        super(RNNBlock, self).__init__()

        self.rnn = nn.LSTM(
            input_size=cfg.rnn_in,
            hidden_size=cfg.rnn_hid,
            num_layers=cfg.rnn_n_layers,
            dropout=cfg.rnn_dropout,
            batch_first=True,
            bidirectional=cfg.rnn_bidirection
        )
        
        cfg.rnn_out = cfg.rnn_hid * cfg.rnn_n_layers * (int(cfg.rnn_bidirection) + 1)
    
    def forward(self, x, x_len):
        b, seq_len, _ = x.size()
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        rnn_output, (hn, cn) = self.rnn(x)
        hn = hn.transpose(0, 1).reshape(b, -1)
        cn = cn.transpose(0, 1).reshape(b, -1)
        return rnn_output, hn, cn
