import hydra
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch import optim
from hydra import utils
from preprocess import preprocess
from dataset import *
from trainer import KFoldTrainer

@hydra.main(config_path='./config.yaml')
def main(cfg):
    # Current Working Directory
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    # Model dictionary
    __MODEL__ = {
        'rnn': 0
    }
    
    # Device
    if cfg.use_GPU and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    
    # Preprocess raw data
    if cfg.preprocess:
        preprocess(cfg)    # Once is quite enough

    # Make dataset
    train_dataset = makeDataset(cfg, train_flag=True)    # len: 100000
    test_dataset = makeDataset(cfg, train_flag=False)    # len: 25000
    
    # Model
    #model = __MODEL__[cfg.model_name].to(device)
    #optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    #criterion = nn.CrossEntropyLoss()
    model, optimizer, criterion = 0, 0, 0
    
    # Train
    k_fold_trainer = KFoldTrainer(cfg, train_dataset, model, device, optimizer, criterion)

    for epoch in range(cfg.EPOCH):
        k_fold_trainer.set_epoch(epoch)
        k_fold_trainer.kFoldTrain()





if __name__ == '__main__':
    main()