import dgl
import torch
import numpy as np
from torch.utils import data
import scipy.sparse as sp

try:
    # for python module
    from .dataset import Dataset
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from dataset import Dataset


class TrainDataset(Dataset, data.dataset.Dataset):

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

    def __getitem__(self, item):
        sid, qid, score = self.raw_data[item]
        concepts = self.concept_map[qid]
        concepts_emb = [0.] * self.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        return sid, qid, torch.Tensor(concepts_emb), score

    def __len__(self):
        return len(self.raw_data)

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
        sek_num = self.n_students + self.n_questions + self.n_concepts
        se_num = self.n_students + self.n_questions
        tmp = np.zeros(shape=(sek_num, sek_num))
        se = np.zeros(shape=(self.n_students, self.n_questions))
        ek = np.zeros(shape=(self.n_questions, self.n_concepts))
        for _, (stu_id, exer_id, label) in enumerate(self._raw_data):
            stu_id, exer_id = int(stu_id), int(exer_id)
            se[stu_id, exer_id] = 1

        for exer_id in self._concept_map:
            for know_id in self._concept_map[exer_id]:
                ek[exer_id, know_id] = 1

        tmp[:self.n_students, self.n_students: se_num] = se
        tmp[self.n_students:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.sp_mat_to_sp_tensor(adj_matrix)
    