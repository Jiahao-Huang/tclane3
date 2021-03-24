import torch
import torch.nn as nn
import numpy as np
from modules import EmbeddingBlock

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        self.model_in = cfg.model_in
        self.attention_hid = cfg.attention_hid
        self.n_heads = cfg.n_heads

        self.Wq = nn.Linear(self.model_in, self.attention_hid * self.n_heads, bias=False)
        self.Wk = nn.Linear(self.model_in, self.attention_hid * self.n_heads, bias=False)
        self.Wv = nn.Linear(self.model_in, self.attention_hid * self.n_heads, bias=False)
        self.fc = nn.Linear(self.attention_hid * self.n_heads, self.model_in, bias=False)


    def forward(self, x, attention_mask):
        batch_size = x.shape[0]

        # batch_size * seq_length * (n_heads * attention_hid) -->  batch_size * n_heads * seq_length * attention_hid
        Q = self.Wq(x).reshape(batch_size, -1, self.n_heads, self.attention_hid).transpose(1, 2)
        K = self.Wk(x).reshape(batch_size, -1, self.n_heads, self.attention_hid).transpose(1, 2)
        V = self.Wv(x).reshape(batch_size, -1, self.n_heads, self.attention_hid).transpose(1, 2)

        A = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.attention_hid)
        A.masked_fill_(attention_mask, -1e9)

        attention = nn.Softmax(dim=-1)(A)

        attention_output = torch.matmul(attention, V)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.attention_hid)
        attention_output = self.fc(attention_output)

        return nn.LayerNorm(self.model_in)(x + attention_output)

class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super(TransformerLayer, self).__init__()
        
        self.model_in = cfg.model_in
        self.ff_hid = cfg.ff_hid

        self.MultiHeadAttention = MultiHeadAttention(cfg)
        self.ff = nn.Sequential(
            nn.Linear(self.model_in, self.ff_hid, bias=False),
            nn.ReLU(),
            nn.Linear(self.ff_hid, self.model_in, bias=False)
        )
        
    
    def forward(self, x, attention_mask):
        attention_output = self.MultiHeadAttention(x, attention_mask)

        ff_output = self.ff(attention_output)
        return nn.LayerNorm(self.model_in)(attention_output + ff_output)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.n_heads = cfg.n_heads
        self.n_layers = cfg.n_layers

        self.embedding = EmbeddingBlock(cfg)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(self.n_layers)]
        )
    
        
    def get_attention_mask(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        attention_mask = x.data.eq(0).unsqueeze(1)
        attention_mask = attention_mask.expand(batch_size, seq_len, seq_len)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        return attention_mask
    
    def forward(self, x):
        attention_mask = self.get_attention_mask(x)
        output = self.embedding(x)
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output, attention_mask)
        return output

        




