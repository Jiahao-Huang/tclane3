import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Embedding

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()

        self.embedding = Embedding(cfg)
        self.kernel_size = cfg.kernel_size
        self.cnn_in_channels = cfg.in_channels
        self.cnn_hid_channels = cfg.cnn_hid_channels
        self.cnn_out_channels = cfg.cnn_out_channels
        self.fc_hid_channels = cfg.fc_hid_channels

        self.conv1 = nn.Conv1d(in_channels=self.cnn_in_channels, out_channels=self.cnn_hid_channels, kernel_size=self.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.cnn_hid_channels, out_channels=self.cnn_out_channels, kernel_size=self.kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.cnn_out_channels, out_channels=self.fc_hid_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(in_channels=self.fc_hid_channels, out_channels=1, kernel_size=1, padding=0)
        self.fc = nn.Linear(self.cnn_out_channels, 1)


    def sent2vec(self, x, len_sent):    # calculate feature vector for a single sentence
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        hidden = self.conv1(x)
        hidden = F.relu(hidden)
        output = self.conv2(hidden)

        alpha = F.relu(output)
        alpha = self.conv3(alpha)
        alpha = F.relu(alpha)
        alpha = self.conv4(alpha)
        alpha = F.softmax(alpha, 2)

        output = output * alpha
        len_sent = len_sent.unsqueeze(1)
        return torch.sum(output, dim=2) / len_sent


    def forward(self, X):
        x = X['x']
        len1 = X['len1']
        len2 = X['len2']

        slen = x.shape[1] // 2

        x1 = x[:, :slen]
        x2 = x[:, slen:]

        output1 = self.sent2vec(x1, len1)
        output2 = self.sent2vec(x2, len2)

        mod1 = torch.sqrt(torch.sum((output1 * output1), dim=1))
        mod2 = torch.sqrt(torch.sum((output2 * output2), dim=1))
        
        output1 = output1 / torch.unsqueeze(mod1, 1)
        output2 = output2 / torch.unsqueeze(mod2, 1)
        
        
        equal = torch.unsqueeze(torch.sum(output1 * output2, 1), 1)
        not_equal = 1 - equal
        return torch.cat((equal, not_equal), 1)
