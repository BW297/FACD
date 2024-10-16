from collections import defaultdict, deque
import torch
import dgl
from torch_geometric.data import Data
import random
import numpy as np
import scipy.sparse as sp

try:
    # for python module
    from .dataset import Dataset
    from .train_dataset import TrainDataset
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from dataset import Dataset
    from train_dataset import TrainDataset


class AdapTestDataset(Dataset):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        super().__init__(data, concept_map,
                         num_students, num_questions, num_concepts)

        # initialize tested and untested set
        self.candidate = None
        self.graph = None
        self.local_map = None
        self.meta = None
        self._tested = None
        self._untested = None
        self.reset()

    def apply_selection(self, student_idx, question_idx):
        """ 
        Add one untested question to the tested set
        Args:
            student_idx: int
            question_idx: int
        """
        assert question_idx in self._untested[student_idx], \
            'Selected question not allowed'
        self._untested[student_idx].remove(question_idx)
        self._tested[student_idx].append(question_idx)
        self.se[student_idx, question_idx] = 1
        self.local_map_new_edge.append((int(student_idx + self.n_questions - 1), int(question_idx)))
        self.inverse_local_map_new_edge.append((int(question_idx), int(student_idx + self.n_questions - 1), ))
    
    def graph_update(self):    
        tmp = np.zeros(shape=(self.sek_num, self.sek_num))
        tmp[:self.n_students, self.n_students: self.se_num] = self.se
        tmp[self.n_students:self.se_num, self.se_num:self.sek_num] = self.ek
        graph = tmp + tmp.T + np.identity(self.sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        self.grpah = self.sp_mat_to_sp_tensor(adj_matrix)

    def local_map_update(self):
        self.local_map['s_from_e'] = dgl.DGLGraph()
        self.local_map['e_from_s'] = dgl.DGLGraph()
        src, dst = tuple(zip(*self.local_map_new_edge))
        self.local_map['s_from_e'].add_edges(src, dst)   
        src, dst = tuple(zip(*self.inverse_local_map_new_edge))
        self.local_map['e_from_s'].add_edges(src, dst)
        self.local_map_new_edge = []
        self.inverse_local_map_new_edge = []

    def reset(self):
        """ 
        Set tested set empty
        """
        self.final_graph()
        self.local_map_new_edge = []
        self.inverse_local_map_new_edge = []
        self.local_map = {
            'k_from_e': self.build_graph4ke(from_e=True),
            'e_from_k': self.build_graph4ke(from_e=False),
            'e_from_s': self.build_graph4se(from_s=True),
            's_from_e': self.build_graph4se(from_s=False),
        }
        self.candidate = dict()
        # self.meta = dict()
        for sid in self.data:
            random.seed(0)
            self.candidate[sid] = self.data[sid].keys()
            # self.candidate[sid] = random.sample(self.data[sid].keys(), int(len(self.data[sid]) * 0.8))
            # self.meta[sid] = [log for log in self.data[sid].keys() if log not in self.candidate[sid]]
        self._tested = defaultdict(deque)
        self._untested = defaultdict(set)
        for sid in self.data:
            self._untested[sid] = set(self.candidate[sid])

    @property
    def tested(self):
        return self._tested

    @property
    def untested(self):
        return self._untested

    def get_tested_dataset(self, last=False,ssid=None,triple=False):
        """
        Get tested data for training
        Args: 
            last: bool, True - the last question, False - all the tested questions
        Returns:
            TrainDataset
        """
        if ssid==None:
            triplets = []
            for sid, qids in self._tested.items():
                if last:
                    qid = qids[-1]
                
                    triplets.append((sid, qid, self.data[sid][qid]))
                else:
                    for qid in qids:
                        triplets.append((sid, qid, self.data[sid][qid]))
            if triple:
                return np.array(triplets)
            else:
                return TrainDataset(triplets, self.concept_map,
                                    self.num_students, self.num_questions, self.num_concepts)
        else:
            triplets = []
            for sid, qids in self._tested.items():
                if ssid == sid:
                    if last:
                        qid = qids[-1]
                        triplets.append((sid, qid, self.data[sid][qid]))
                    else:
                        for qid in qids:
                            triplets.append((sid, qid, self.data[sid][qid]))
            return TrainDataset(triplets, self.concept_map,
                                self.num_students, self.num_questions, self.num_concepts)
        
    def get_meta_dataset(self):
        triplets = {}
        for sid, qids in self.meta.items():
            triplets[sid] = {}
            for qid in qids:
                triplets[sid][qid] = self.data[sid][qid]
        return triplets
    
    def build_graph4se(self, from_s: bool):
        stu_num = self.n_students
        exer_num = self.n_questions
        node = stu_num + exer_num
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        cnt = {}
        for stu in range(self.num_students):
            cnt[stu] = 0

        for _, (stu_id, exer_id, label) in enumerate(self._raw_data):
            if cnt[stu] > 30:
                continue
            cnt[stu] += 1
            if from_s:
                edge_list.append((int(stu_id + exer_num - 1), int(exer_id)))
            else:
                edge_list.append((int(exer_id), int(stu_id + exer_num - 1)))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    
    def build_graph4ke(self, from_e: bool):
        know_num = self.n_concepts
        exer_num = self.n_questions
        node = exer_num + know_num
        g = dgl.DGLGraph()
        g.add_nodes(node)
        edge_list = []
        if from_e:
            for exer_id in self._concept_map:
                for know_id in self._concept_map[exer_id]:
                    edge_list.append((int(exer_id), int(know_id + exer_num - 1)))
        else:
            for exer_id in self._concept_map:
                for know_id in self._concept_map[exer_id]:
                    edge_list.append((int(know_id + exer_num - 1), int(exer_id)))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    
    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()
    
    def final_graph(self):
        self.sek_num = self.n_students + self.n_questions + self.n_concepts
        self.se_num = self.n_students + self.n_questions
        tmp = np.zeros(shape=(self.sek_num, self.sek_num))
        self.se = np.zeros(shape=(self.n_students, self.n_questions))
        self.ek = np.zeros(shape=(self.n_questions, self.n_concepts))
        for _, (stu_id, exer_id, label) in enumerate(self._raw_data):
            stu_id, exer_id = int(stu_id), int(exer_id)
            self.se[stu_id, exer_id] = 1

        for exer_id in self._concept_map:
            for know_id in self._concept_map[exer_id]:
                self.ek[exer_id, know_id] = 1

        tmp[:self.n_students, self.n_students: self.se_num] = self.se
        tmp[self.n_students:self.se_num, self.se_num:self.sek_num] = self.ek
        graph = tmp + tmp.T + np.identity(self.sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        self.graph = self.sp_mat_to_sp_tensor(adj_matrix)