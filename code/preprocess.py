import os
import csv
import pickle
from torch import tensor


def tsv2pkl(tsv_f, pkl_f, train_flag):  # .tsv => .pkl
    pkl_list = []
    with open(tsv_f) as f:
        tsv_data = csv.reader(f, delimiter='\t')
        for l in tsv_data:
            pkl_dict = {}
            pkl_dict['x1'] = [int(x) for x in l[0].split(' ')] 
            pkl_dict['x2'] = [int(x) for x in l[1].split(' ')]
            pkl_dict['len1'] = len(pkl_dict['x1'])
            pkl_dict['len2'] = len(pkl_dict['x2'])
            if train_flag: 
                pkl_dict['y'] = int(l[2])
            pkl_list.append(pkl_dict)

    with open(pkl_f, 'wb') as f:
        pickle.dump(pkl_list, f)
    


def preprocess(cfg, swap=True):
    # train_tsv = "../tcdata/oppo_breeno_round1_data/train.tsv"
    # test_tsv = "../tcdata/oppo_breeno_round1_data/testB.tsv"
    # train_tmp = "../user_data/tmp_data/train.pkl"
    # test_tmp = "../user_data/tmp_data/test.pkl"

    train_tsv = os.path.join(cfg.cwd, cfg.train_tsv)
    test_tsv = os.path.join(cfg.cwd, cfg.test_tsv)
    train_tmp = os.path.join(cfg.cwd, cfg.train_tmp)
    test_tmp = os.path.join(cfg.cwd, cfg.test_tmp)
    
    tsv2pkl(train_tsv, train_tmp, train_flag=True)
    tsv2pkl(test_tsv, test_tmp, train_flag=False)

if __name__ == "__main__":
    pass