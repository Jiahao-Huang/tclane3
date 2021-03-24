import random
import torch
import logging

from dataset import collate_fn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

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

        self.data_size = cfg.data_size_train
        self.fold_size = cfg.data_size_train / self.k_fold
        self.data_size_train = self.fold_size * (self.k_fold - 1)

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
        tot_data = 0
        tot_loss = 0
        tot_correct = 0
        percent = -1

        for (X, y) in train_dataloader:
            for (k, v) in X.items():
                X[k] = v.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X)

            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                tot_loss += loss
                tot_data += y.shape[0]
                tot_correct += torch.eq(torch.argmax(y_pred, dim=1), y).sum().item()
                if percent != (tot_data * 10) // self.data_size_train:
                    percent = (tot_data * 10) // self.data_size_train
                    logger.info("[%d%%] Train Loss: %.6f Accuracy: %.2f%%"%(tot_data * 100 // self.data_size_train, tot_loss / tot_data, tot_correct / tot_data*100))

        return tot_data, tot_loss, tot_correct

    def validate(self, valid_dataloader):
        tot_data = 0
        tot_loss = 0
        tot_correct = 0

        for (X, y) in valid_dataloader:
            with torch.no_grad():
                for (k, v) in X.items():
                    X[k] = v.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(X)

                loss = self.criterion(y_pred, y)

                tot_loss += loss
                tot_data += y.shape[0]
                tot_correct += torch.eq(torch.argmax(y_pred, dim=1), y).sum().item()
        
        logger.info("Valid Loss: %.6f Accuracy: %.2f%%"%(tot_loss / tot_data, tot_correct / tot_data*100))
        return tot_data, tot_loss, tot_correct
    
    def kFoldTrain(self):
        cfg = self.cfg

        # make folds
        self.train_dataset.shuffle()
        folds = self.make_folds()
        for k in range(self.k_fold):   # fold-k will be valid set
            logger.info("="*5 + " Epoch %d - Fold %d " % (self.epoch, k) + "="*5)
            train = []
            valid = folds[k]
            for i in range(self.k_fold):    # folds except k will be train set
                if i != k:
                    train = train + folds[i]
            
            train_dataloader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
            valid_dataloader = DataLoader(valid, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

            self.train(train_dataloader)
            self.validate(valid_dataloader)



            

