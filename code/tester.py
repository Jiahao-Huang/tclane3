import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import collate_fn
import os


class Tester:
    def __init__(self, cfg, test_dataset, model, device):
        self.cfg = cfg
        self.device = device
        self.test_dataset = test_dataset

        if cfg.test_with_best_model:
            self.model = torch.load(os.path.join(cfg.cwd, "{}model_{}.pt".format(cfg.model_dir, cfg.model_name))).to(self.device)
        else:
            self.model = model.to(self.device)
        model.eval()
        

    def test(self):
        cfg = self.cfg

        test_dataloader = DataLoader(self.test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn(cfg))
        results = []

        for (X, _) in test_dataloader:
            with torch.no_grad():
                for (k, v) in X.items():
                    X[k] = v.to(self.device)

                y_pred = self.model(X)

                result = F.softmax(y_pred, dim=-1)[:, 1].to('cpu').tolist()
                results += result
        
        with open(os.path.join(cfg.cwd, cfg.result_file), 'w') as f:
            f.write('\t'.join('%s'%r for r in results))
