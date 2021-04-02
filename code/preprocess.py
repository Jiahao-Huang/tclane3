import os
import csv
import pickle
import json
from transformers import BertTokenizer
import numpy as np


def tsv2pkl(tsv_f, pkl_f, tokens, train_flag):  # .tsv => .pkls
    pkl_list = []
    with open(tsv_f) as f:
        tsv_data = csv.reader(f, delimiter='\t')
        data_size = 0
        for l in tsv_data:
            pkl_dict = {}
            x1 = [int(x) for x in l[0].split(' ')] 
            x2 = [int(x) for x in l[1].split(' ')]
            
            for word in x1 + x2:
                tokens[word] = tokens.get(word, 0) + 1

            pkl_dict['x1'] = x1
            pkl_dict['x2'] = x2
            pkl_dict['len1'] = len(x1)
            pkl_dict['len2'] = len(x2)
            if train_flag: 
                pkl_dict['y'] = int(l[2])
            
            pkl_list.append(pkl_dict)
            data_size += 1
        
    with open(pkl_f, 'wb') as f:
        pickle.dump(pkl_list, f)
    
    return tokens, data_size


def tokenTransfer(pkl_f, tokens, bert_tokens):
    with open(pkl_f, "rb") as f:
        data = pickle.load(f)
    for d in data:    # d: {x1, x2, len1, len2}
        for word in d['x1'] + d['x2']:
            word = bert_tokens[tokens.get(word, 1)]
        d['x1'] = [101] + d['x1'] + [102]
        d['x2'] = [101] + d['x2'] + [102]
        d['len1'] = len(d['x1'])
        d['len2'] = len(d['x2'])
    with open(pkl_f, 'wb') as f:
        pickle.dump(data, f)


def preprocess(cfg):
    # train_tsv = "../tcdata/oppo_breeno_round1_data/train.tsv"
    # test_tsv = "../tcdata/oppo_breeno_round1_data/testB.tsv"
    # train_tmp = "../user_data/tmp_data/train.pkl"
    # test_tmp = "../user_data/tmp_data/test.pkl"

    train_tsv = os.path.join(cfg.cwd, cfg.train_tsv)
    test_tsv = os.path.join(cfg.cwd, cfg.test_tsv)
    train_tmp = os.path.join(cfg.cwd, cfg.train_tmp)
    test_tmp = os.path.join(cfg.cwd, cfg.test_tmp)
    
    tokens = {}
    
    tokens, data_size_train = tsv2pkl(train_tsv, train_tmp, tokens, train_flag=True)
    tokens, data_size_test = tsv2pkl(test_tsv, test_tmp, tokens, train_flag=False)

    cfg.data_size_train = data_size_train
    cfg.data_size_test = data_size_test
    # cfg.vocab_size = len(tokens)

    # Initialization for bert
    if cfg.model_name == 'bert':
        # sort tokens in training data
        tokens = {i: j for i, j in tokens.items() if j >= cfg.min_freq}
        tokens = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
        tokens = {t[0]: i+7 for i, t in enumerate(tokens)}    # word to index

        # word frequency of bert 
        with open(os.path.join(cfg.cwd, cfg.bert_freq), encoding="utf-8") as f:
            bert_freq = json.load(f)
        with open(os.path.join(cfg.cwd, cfg.bert_vocab), encoding="utf=8") as f:
            bert_vocab = [l.strip() for l in f.readlines()]
        
        del bert_freq['[CLS]']
        del bert_freq['[SEP]']
        freq = [bert_freq.get(word, 0) for (i, word) in enumerate(bert_vocab)]
        # sort tokens in bert
        bert_tokens = list(np.argsort(freq)[::-1])
        bert_tokens = [0, 100, 101, 102, 103, 100, 100] + bert_tokens[:len(tokens)]

        tokenTransfer(train_tmp, tokens, bert_tokens)
        tokenTransfer(test_tmp, tokens, bert_tokens)



if __name__ == "__main__":
    pass