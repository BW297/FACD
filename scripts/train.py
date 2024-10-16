import sys
import os
import argparse
import torch
sys.path.append('..')
from dataset.train_dataset import TrainDataset
from model.IRT import IRTModel
from model.NCD import NCDModel
from model.FACD import FACDModel
import json
import random
import logging
import numpy as np
import pandas as pd

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FrcSub', type=str, help='benchmark')
parser.add_argument('--cdm', default='rcd', type=str)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--decoder', default='ncd', type=str)
parser.add_argument('--test_size', default=0.8, type=float)
parser.add_argument('--out_channels', default=32, type=int)
parser.add_argument('--seed', default=0, type=int)

parser = parser.parse_args()

if __name__ == '__main__':
    set_seed(parser.seed)
    dataset = parser.dataset
    # modify config here
    config = {
        'cdm': parser.cdm,
        'learning_rate': parser.lr,
        'batch_size': parser.batch_size,
        'num_epochs': parser.num_epochs,
        'num_dim': 1, # for IRT or MIRT
        'device': parser.device,
        'test_size': parser.test_size,
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
        'betas': (0.9, 0.999),
        # for GrpahCD
        'decoder': parser.decoder,
        'out_channels': parser.out_channels,
        'num_layers': 2,
        'weight_reg': 0
    }

    # read datasets
    response_logs = pd.read_csv(f'../data/{dataset}/TotalData.csv', header=None)
    q = pd.read_csv(f'../data/{dataset}/q.csv', header=None)
    from data.data_params_dict import data_params
    metadata = {
        'stu_num': data_params[dataset]['stu_num'],
        'prob_num': data_params[dataset]['prob_num'],
        'know_num': data_params[dataset]['know_num'],
    }
    config['q'] = q
    config['stu_num'] = metadata['stu_num']
    config['prob_num'] = metadata['prob_num']
    config['know_num'] = metadata['know_num']
    total_stu = list(range(0, metadata['stu_num']))
    train_stu = random.sample(total_stu, int(len(total_stu) * config['test_size']))
    pre_train_stu = [stu for stu in total_stu if stu not in train_stu]
    train_triplets = []
    pre_train_triplets = []
    concept_map = {}
    cnt = 0
    for log in response_logs.values:
        if int(log[0]) in train_stu:
            train_triplets.append((int(log[0]), int(log[1]), int(log[2])))
        else:
            pre_train_triplets.append((int(log[0]), int(log[1]), int(log[2])))
    for question in q.values:
        concept_map[cnt] = np.where(np.array(question))[0].tolist()
        cnt += 1

    train_data = TrainDataset(train_triplets, concept_map,
                                        metadata['stu_num'], 
                                        metadata['prob_num'], 
                                        metadata['know_num'])
    pre_train_data = TrainDataset(pre_train_triplets, concept_map,
                                        metadata['stu_num'], 
                                        metadata['prob_num'], 
                                        metadata['know_num'])
    config['pre_train_triplets'] = np.array(pre_train_triplets)

    # define model here
    if config['cdm'] == 'irt':
        model = IRTModel(**config)
    elif config['cdm'] == 'ncd':
        model = NCDModel(**config)
    else:
        config['graph'] = pre_train_data.final_graph()
        model = FACDModel(**config)
    # train model
    model.init_model(pre_train_data)
    model.train(pre_train_data)
    model.adaptest_save('../model/ckpt/{}_{}.pt'.format(dataset, config['cdm']))



