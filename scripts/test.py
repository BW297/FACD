import sys
import os
import torch
sys.path.append('..')
import argparse
import torch
import numpy as np
import pandas as pd
import random
from strategy.NCAT_strategy import NCATs
from strategy.random_strategy import RandomStrategy
from strategy.MAAT_strategy import MAATStrategy
from strategy.BECAT_strategy import BECATstrategy
from model.IRT import IRTModel
from model.NCD import NCDModel
from model.FACD import FACDModel
from dataset.adaptest_dataset import AdapTestDataset

import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument('--strategy', default='random', type=str)
parser.add_argument('--cdm', default='graph', type=str)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--decoder', default='ncd', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--test_size', default=0.8, type=float)
parser.add_argument('--test_length', default=15, type=int)
parser.add_argument('--out_channels', default=32, type=int)

parser = parser.parse_args()

if __name__ == '__main__':
    set_seed(parser.seed)
    # modify config here
    config = {
        'dataset': parser.dataset,
        'learning_rate': parser.lr,
        'batch_size': parser.batch_size,
        'num_epochs': parser.num_epochs,
        'num_dim': 1,
        'device': parser.device,
        'cdm': parser.cdm,
        'seed': parser.seed,
        'test_size': parser.test_size,
        'test_length': parser.test_length,
        'strategy': parser.strategy,
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
        # for NCAT
        'THRESHOLD' :300,
        'start':0,
        'end':3000,
        'betas': (0.9, 0.999),
        # for GrpahCD
        'decoder': parser.decoder,
        'out_channels': parser.out_channels,
        'num_layers': 2,
        'weight_reg': 0
    }
    # fixed test length
    test_length = config['test_length']
    # choose strategies here

    if parser.strategy == 'random':
        strategies = [RandomStrategy()]
    elif parser.strategy == 'MAAT':
        strategies = [MAATStrategy()]
    elif parser.strategy == 'NCAT':
        strategies = [NCATs()]
    elif parser.strategy == 'BECAT':
        strategies = [BECATstrategy()]

    # modify checkpoint path here
    ckpt_path = f'../model/ckpt/{parser.dataset}_{parser.cdm}.pt'
    # read datasets
    response_logs = pd.read_csv(f"../data/{config['dataset']}/TotalData.csv", header=None)
    q = pd.read_csv(f"../data/{config['dataset']}/q.csv", header=None)
    from data.data_params_dict import data_params
    metadata = {
        'stu_num': data_params[config['dataset']]['stu_num'],
        'prob_num': data_params[config['dataset']]['prob_num'],
        'know_num': data_params[config['dataset']]['know_num'],
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

    train_data = AdapTestDataset(train_triplets, concept_map,
                                        metadata['stu_num'], 
                                        metadata['prob_num'], 
                                        metadata['know_num'])
    pre_train_data = AdapTestDataset(pre_train_triplets, concept_map,
                                       metadata['stu_num'], 
                                       metadata['prob_num'],
                                       metadata['know_num'])
    config['pre_train_triplets'] = np.array(pre_train_triplets)
    for strategy in strategies:
        avg =[]
        train_data.reset()
        pre_train_data.final_graph()
        train_data.graph = pre_train_data.graph
        train_data.se = pre_train_data.se
        train_data.ek = pre_train_data.ek
        if parser.cdm == 'irt':
            model = IRTModel(**config)
        elif parser.cdm == 'ncd':
            model = NCDModel(**config)
        else:
            model = FACDModel(**config)

        model.init_model(train_data)
        model.adaptest_load(ckpt_path)
        print(strategy.name)
        if strategy.name == 'NCAT':
            selected_questions = strategy.adaptest_select(train_data,concept_map,config,test_length)
            for it in range(test_length):
                for student, questions in selected_questions.items():
                    train_data.apply_selection(student, questions[it])  
                model.adaptest_update(train_data)
                results = model.evaluate(train_data)
                print(results)
            continue
        S_sel ={}
        for sid in train_data.data.keys():
            key = sid
            S_sel[key] = []
        selected_questions={}
        total = 0
        for it in range(1, test_length + 1):
            # select question
            model.config['it'] = it
            if it == 1 and strategy.name == 'BECAT Strategy':
                for sid in train_data.data.keys():
                    untested_questions = np.array(list(train_data.untested[sid]))
                    random_index = random.randint(0, len(untested_questions)-1)
                    selected_questions[sid] = untested_questions[random_index]
                    S_sel[sid].append(untested_questions[random_index])
            elif strategy.name == 'BECAT Strategy':    
                selected_questions = strategy.adaptest_select(model, train_data, S_sel)
                for sid in train_data.data.keys():
                    S_sel[sid].append(selected_questions[sid])
            else:
                selected_questions = strategy.adaptest_select(model, train_data)
            for student, question in selected_questions.items():
                train_data.apply_selection(student, question)
            train_data.graph_update()    
            
            # update models
            model.adaptest_update(train_data)
            # evaluate models
            results = model.evaluate(train_data)
            print(results)
