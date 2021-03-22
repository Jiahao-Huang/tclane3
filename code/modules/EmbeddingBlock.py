import torch
import torch.nn as nn

class EmbeddingBlock(nn.Module):
    def __init__(self, cfg):
        super(EmbeddingBlock, self).__init__()

        self.vocab_size = cfg.vocab_size
        self.embedding_dim = cfg.embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        # self.layernorm = nn.LayerNorm(self.embedding_dim)

        cfg.model_in = self.embedding_dim


    def forward(self, x):    # batch_size * xlen
        x = self.embedding(x)
        return x    # batch_size * xlen * embedding_dim





