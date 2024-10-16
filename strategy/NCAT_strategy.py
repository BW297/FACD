from math import exp as exp
from sklearn.metrics import roc_auc_score
from dataset import AdapTestDataset
from sklearn.metrics import accuracy_score
from collections import namedtuple
from scipy.optimize import minimize
from strategy.abstract_strategy import AbstractStrategy
from strategy.NCAT_nn.NCAT import NCATModel
from tqdm import tqdm
import numpy as np
import random

class NCATs(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'NCAT'

    def adaptest_select(self, adaptest_data: AdapTestDataset,concept_map,config,test_length):
        selection = {}
        NCATdata = adaptest_data
        model = NCATModel(NCATdata,concept_map,config,test_length)
        threshold = config['THRESHOLD']
        total_stu = adaptest_data.data.keys()
        for sid in tqdm(total_stu, "Policy Selecting: "):
            # print(str(sid+1)+'/'+str(adaptest_data.num_students))
            used_actions = []
            model.ncat_policy(sid,threshold,used_actions,type="training",epoch=100)
        NCATdata.reset()
        for sid in tqdm(total_stu, "Evaluating Policy: "):
            used_actions = []
            model.ncat_policy(sid,threshold,used_actions,type="testing",epoch=0)
            selection[sid] = used_actions
        NCATdata.reset()
        return selection
