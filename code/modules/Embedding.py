import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, cfg):
        super(Embedding, self).__init__()

        self.vocab_size = cfg.vocab_size
        self.embedding_dim = cfg.embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.layernorm = nn.LayerNorm(self.embedding_dim * 2)


    def forward(self, x):    # batch_size * xlen
        xlen = x.shape[1]
        slen = xlen // 2

        x1 = x[:, :slen]
        x2 = x[:, slen+1:]

        word_embedding1 = self.embedding(x1)
        word_embedding2 = self.embedding(x2)

        word_embedding = torch.cat((word_embedding1, word_embedding2), 2)

        return self.layernorm(word_embedding)





