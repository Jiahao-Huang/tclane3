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

    def inner_product(self, y1, y2):
        y1 = y1.unsqueeze(-1)
        y2 = y2.unsqueeze(-1)
        res = torch.matmul(y1.transpose(1, 2), y2)
        return res.view(-1, 1)
            
    def train(self, train_dataloader):
        tot_loss = 0
        len_data = 0

        for (X1, X2, y) in train_dataloader:
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y1_pred = self.model(X1)
            y2_pred = self.model(X2)
            similarity = self.inner_product(y1_pred, y2_pred) / self.inner_product(y1_pred, y1_pred) / self.inner_product(y2_pred, y2_pred)

            loss = self.criterion(torch.cat((-similarity, similarity), dim=1), y)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                l = X1.shape[0]
                len_data = len_data + l
                tot_loss = tot_loss + l * loss
        
        avg_loss = tot_loss / len_data
        return avg_loss

    def validate(self, valid_dataloader):
        for (X1, X2, y) in valid_dataloader:
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

            avg_loss = self.train(train_dataloader)
            print("TRAINING: Epoch%d.Fold%d. Average Loss:%f " %(self.epoch, k, avg_loss))
            #self.validate(valid_dataloader)



            

