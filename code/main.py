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
from tester import Tester

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
        'rnn': models.RNN,
        'cnn':models.CNN,
        'bert':models.Bert
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
    logger.info("Model:{}".format(cfg.model_name))

    model = __MODEL__[cfg.model_name](cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    if not cfg.test_only:
        logger.info("="*5 + " Start %d-fold training! "%cfg.k_fold + "="*5)

        k_fold_trainer = KFoldTrainer(cfg, train_dataset, model, device, optimizer, criterion)

        best_epoch, best_acc_v = -1, -1
        
        for epoch in range(cfg.EPOCH):
            k_fold_trainer.set_epoch(epoch)
            loss_t, acc_t, loss_v, acc_v = k_fold_trainer.kFoldTrain()
            if (acc_v > best_acc_v):
                best_epoch = epoch
                best_acc_v = acc_v
                torch.save(model, os.path.join(cwd, "{}model_{}.pt".format(cfg.model_dir, cfg.model_name)))
                logger.info("Model Updated!")

            logger.info("Current Epoch(%d):\nTrain loss: %.6f Train accuracy: %.2f%% Valid loss: %.6f Valid Accuracy: %.2f%%"%(epoch, loss_t, acc_t*100, loss_v, acc_v*100))
            logger.info("Best Epoch(%d):\n Accuracy: %.2f%%"%(best_epoch, best_acc_v * 100))

        
        logger.info("="*5 + " Training finished. " + "="*5)

    # Test
    logger.info("="*5 + " Start testing! " + "="*5)
    tester = Tester(cfg, test_dataset, model, device)
    tester.test()
    logger.info("Saving results to {}. ".format(os.path.join(cwd, "{}".format(cfg.result_file))))
    logger.info("="*5 + " Test finished. " + "="*5)


if __name__ == '__main__':
    main()