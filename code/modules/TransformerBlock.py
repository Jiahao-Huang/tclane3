import torch
import torch.nn as nn
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.transformer_in, nhead=cfg.transformer_n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.transformer_n_layers)
        cfg.transformer_out = cfg.transformer_in
    
    def forward(self, x):
        return self.encoder(x)


'''
def get_attn_pad_mask(seq_q, seq_k):
    b, len_q = seq_q.size()
    b, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)    # batch_size * 1 * len_k
    return pad_attn_mask.expand(b, len_q, len_k)


class PositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super(PositionalEncoding, self).__init__()
        self.max_len = cfg.pos_enc_max_len
        self.model_in = cfg.model_in

        self.dropout = nn.Dropout(cfg.pos_enc_dropout)


        pe = torch.zeros(self.max_len, self.model_in)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_in, 2, dtype=torch.float) * (-math.log(10000.0) / self.model_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: seq_len * batch_size * model_in
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, cfg):
        self.attention_hid = cfg.attention_hid
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, attn_mask):    
        # Q, K, V: batch_size * n_heads * seq_len * attention_hid
        # attn_mask: batch_size * n_heads * seq_len * seq_len
        A = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.attention_hid)    # A: batch_size * n_heads * seq_len * seq_len
        A.masked_fill_(attn_mask, -1e9)

        A = nn.Softmax(dim=-1)(A)
        attention_output = torch.matmul(A, V)    # attention_output: batch_size * n_heads * seq_len * attention_hid
        return attention_output, A
    

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        self.cfg = cfg
        self.model_in = cfg.model_in
        self.attention_hid = cfg.attention_hid
        self.n_heads = cfg.n_heads
        self.Wq = nn.Linear(self.model_in, self.attention_hid * self.n_heads, bias=False)
        self.Wk = nn.Linear(self.model_in, self.attention_hid * self.n_heads, bias=False)
        self.Wv = nn.Linear(self.model_in, self.attention_hid * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.attention_hid, self.model_in, bias=False)
    
    def forward(self, q, k, v, attn_mask):
        residual = q
        b, seq_len, _ = q.size()

        Q = self.Wq(q).reshape(b, -1, self.n_heads, self.attention_hid).transpose(1, 2)
        K = self.Wk(k).reshape(b, -1, self.n_heads, self.attention_hid).transpose(1, 2)
        V = self.Wv(v).reshape(b, -1, self.n_heads, self.attention_hid).transpose(1, 2)

        # attn_mask: batch_size * seq_len * seq_len => batch_size * n_heads * seq_len * seq_len
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        attention_output, A = ScaledDotProductAttention(self.cfg)(Q, K, V, attn_mask)    
        # attention_output: batch_size * n_heads * seq_len * attention_hid => batch_size * seq_len * attention_hid_all
        attention_output = attention_output.transpose(1, 2).reshape(b, -1, self.attention_hid * self.n_heads)
        # print(attention_output.shape)
        # print(self.fc)
        attention_output = self.fc(attention_output)

        return nn.LayerNorm(self.model_in)(residual + attention_output), A


class FeedForward(nn.Module):
    def __init__(self, cfg):
        self.model_in = cfg.model_in
        self.ff_hid = cfg.ff_hid
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(self.model_in, self.ff_hid), 
            nn.ReLU(),
            nn.Linear(self.ff_hid, self.model_in)
        )

    def forward(self, x):
        residual = x
        output = self.fc(x)
        return nn.LayerNorm(self.model_in)(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(cfg)
        self.feed_forward = FeedForward(cfg)

    def forward(self, enc_input, attn_mask):
        attention_output, A = self.attention(enc_input, enc_input, enc_input, attn_mask)
        enc_output = self.feed_forward(attention_output)
        return enc_output, A

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.vocab_size = cfg.vocab_size
        self.model_in = cfg.model_in
        self.n_layers = cfg.n_layers

        self.embedding = nn.Embedding(self.vocab_size, self.model_in, padding_idx=0)
        self.pos_encoding = PositionalEncoding(cfg)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(self.n_layers)])

    def forward(self, enc_input):
        embed_output = self.embedding(enc_input)
        enc_output = self.pos_encoding(embed_output.transpose(0, 1)).transpose(0, 1)
        self_attn_mask = get_attn_pad_mask(enc_input, enc_input)
        for layer in self.layers:
            enc_output, _ = layer(enc_output, self_attn_mask)
        return torch.sum(enc_output, dim=-1)
'''