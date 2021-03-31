import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, cfg):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv1d(cfg.model_in, cfg.cnn_out, cfg.kernal_size, padding=cfg.padding)
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.cnn_out, cfg.fc_hid),
            nn.ReLU(),
            nn.Linear(cfg.fc_hid, cfg.cnn_out)
        )
        self.weight_fc = nn.Linear(cfg.cnn_out, cfg.cnn_out)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        cnn_output = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        residual = cnn_output
        cnn_output = nn.ReLU()(cnn_output)
        ff_output = self.feed_forward(cnn_output.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)

        output = residual + ff_output

        weight = self.weight_fc(cnn_output.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
        weight = nn.Softmax(dim=-1)(weight)*10000

        weighted_output = output * weight
        return output, weighted_output

        
