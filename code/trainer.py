import random
import torch

from dataset import collate_fn
from torch.utils.data import DataLoader

class KFoldTrainer:
    def __init__(self, cfg, train_dataset, model, device, optimizer, criterion):
        self.epoch = 0

        self.cfg = cfg
        self.k_fold = cfg.k_fold
        self.train_dataset = train_dataset
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.data_size = len(train_dataset)
        self.fold_size = len(train_dataset) / self.k_fold

    def set_epoch(self, e):
        self.epoch = e

    def make_folds(self):
        fold_id = -1
        folds = []
        for data_id in range(self.data_size):
            if not data_id % self.fold_size:
                folds.append([])
                fold_id += 1
            folds[fold_id].append(self.train_dataset[data_id])
        return folds


    def train(self, train_dataloader):
        for (X, y) in train_dataloader:
            for key, value in X.items():
                X[key] = value.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        return 0

    def validate(self, valid_dataloader):
        for (X, y) in valid_dataloader:
            with torch.no_grad():
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
        return 0
    
    def kFoldTrain(self):
        cfg = self.cfg

        # make folds
        self.train_dataset.shuffle()
        folds = self.make_folds()
        for k in range(self.k_fold):   # fold-k will be valid set
            train = []
            valid = folds[k]
            for i in range(self.k_fold):    # folds except k will be train set
                if i != k:
                    train = train + folds[i]
            
            train_dataloader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
            valid_dataloader = DataLoader(valid, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

            self.train(train_dataloader)
            self.validate(valid_dataloader)



            

