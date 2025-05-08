from collections import defaultdict, deque
import torch
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
        self.meta = None
        self._tested = None
        self._untested = None
        self.se = None
        self.ek = None
        self.sek_num = self.n_students + self.n_questions + self.n_concepts
        self.se_num = self.n_students + self.n_questions
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
        self.graph = self.sp_mat_to_sp_tensor(adj_matrix)

    def reset(self):
        """ 
        Set tested set empty
        """
        self.candidate = dict()
        for sid in self.data:
            self.candidate[sid] = self.data[sid].keys()
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
    
    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()
    
    def final_graph(self):
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