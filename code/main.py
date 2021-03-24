import hydra
import logging
import torch
import torch.nn as nn
import models

from torch.utils.data import Dataset, DataLoader
from torch import optim
from hydra import utils
from preprocess import preprocess
from dataset import *
from trainer import KFoldTrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path='config/config.yaml')
def main(cfg):
    logger.info("=" * 20)
    # Current Working Directory
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    # Model dictionary
    __MODEL__ = {
        'transformer': models.Transformer,
        'rnn': models.RNN
    }
    
    # Device     
    if cfg.use_GPU and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info("Device: {}".format(device))

    # Preprocess raw data
    if cfg.preprocess:
        logger.info("Preprocessing...")
        preprocess(cfg)    # Once is quite enough
    
    # Make dataset
    logger.info("Making Dataset...")
    train_dataset = makeDataset(cfg, train_flag=True)    # len: 100000
    test_dataset = makeDataset(cfg, train_flag=False)    # len: 25000
    
    # Model
    model = __MODEL__[cfg.model_name](cfg).to(device)
    logger.info("Model:{}".format(cfg.model_name))

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    logger.info("="*5 + " Start %d-fold training! "%cfg.k_fold + "="*5)
    k_fold_trainer = KFoldTrainer(cfg, train_dataset, model, device, optimizer, criterion)
 
    for epoch in range(cfg.EPOCH):
        k_fold_trainer.set_epoch(epoch)
        k_fold_trainer.kFoldTrain()





if __name__ == '__main__':
    main()