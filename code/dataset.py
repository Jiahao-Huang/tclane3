import os
import pickle
import torch

from random import randint, shuffle
from torch.utils.data import Dataset

class pklDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data_tmp = pickle.load(f)
    
    def __len__(self):
        return len(self.data_tmp)
    
    def __getitem__(self, item):
        return self.data_tmp[item]
    
    def shuffle(self):
        shuffle(self.data_tmp)


def makeDataset(cfg, train_flag):
    if train_flag:
        pkl_path = os.path.join(cfg.cwd, cfg.train_tmp)
    else:
        pkl_path = os.path.join(cfg.cwd, cfg.test_tmp)
    
    return pklDataset(pkl_path)

def collate_fn(cfg):
    def get_max_len(batch):
        max_len = 0
        for b in batch:
            max_len = max(max(b['len1'], b['len2']), max_len)
        return max_len
    
    def padding(x, max_len):
        return x + [0] * (max_len - len(x))

    # collate function
    def collate_fn_b(batch):
        max_len = get_max_len(batch)
        X = {}
        X1, X2, y = [], [], []
        len1, len2 = [], []
        for b in batch:
            x1 = padding(b['x1'], max_len)
            x2 = padding(b['x2'], max_len)
            if cfg.swap and randint(0,1):    # randomly swap
                x1, x2 = x2, x1
                b['len1'], b['len2']  = b['len2'], b['len1']
            X1.append(x1)
            X2.append(x2)
            len1.append(b['len1'])
            len2.append(b['len2'])
            if 'y' in b:
                y.append(b['y'])

        X['X1'] = torch.tensor(X1)
        X['X2'] = torch.tensor(X2)
        X['len1'] = torch.tensor(len1)
        X['len2'] = torch.tensor(len2)
        
        y = torch.tensor(y)
        return X, y
    
    return collate_fn_b
            
            

