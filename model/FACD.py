import torch
import math
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_sparse import SparseTensor
from model.abstract_model import AbstractModel
from .utils import StraightThrough, create_dncoder
from sklearn.metrics import roc_auc_score,accuracy_score,mean_squared_error, f1_score

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, gcn_layers=1, gcn_drop=True, keep_prob=0.7):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gcn_layers = gcn_layers
        self.gcn_drop = gcn_drop
        self.keep_prob = keep_prob

    def convolution(self, graph, all_emb):
        emb = [self.fc(all_emb)]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.graph_drop(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb
    
    def dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def graph_drop(self, graph):
        g_dropped = self.dropout(graph, self.keep_prob)
        return g_dropped

class AdditiveAttention(nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.query_vector_dim = query_vector_dim
        self.candidate_vector_dim = candidate_vector_dim
        
        # Linear transformation to project candidate vectors to query vector space
        self.dense = nn.Linear(candidate_vector_dim, query_vector_dim)
        
        # Learnable query vector
        self.attention_query_vector = nn.Parameter(torch.randn(query_vector_dim, 1) * 0.1)

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: Tensor of shape (batch_size, candidate_size, candidate_vector_dim)
        Returns:
            A Tensor of shape (batch_size, candidate_vector_dim)
        """
        # Apply dense layer and tanh activation
        dense_output = torch.tanh(self.dense(candidate_vector))  # Shape: (batch_size, candidate_size, query_vector_dim)

        # Compute the attention scores by multiplying with the query vector
        candidate_weights = torch.matmul(dense_output, self.attention_query_vector).squeeze(-1)  # Shape: (batch_size, candidate_size)

        # Apply softmax to normalize the attention scores
        candidate_weights = F.softmax(candidate_weights, dim=1)  # Shape: (batch_size, candidate_size)

        # Compute the weighted sum of candidate vectors
        candidate_weights = candidate_weights.unsqueeze(1)  # Shape: (batch_size, 1, candidate_size)
        target = torch.bmm(candidate_weights, candidate_vector).squeeze(1)  # Shape: (batch_size, candidate_vector_dim)

        return target


# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        # Q: [batch_size, num_heads, candidate_num, d_k]
        # K: [batch_size, num_heads, candidate_num, d_k]
        # V: [batch_size, num_heads, candidate_num, d_v]
        
        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch_size, num_heads, candidate_num, candidate_num]
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)  # Apply mask
        
        # Softmax over the last dimension (candidate_num)
        attn = F.softmax(scores, dim=-1)  # [batch_size, num_heads, candidate_num, candidate_num]
        
        # Weighted sum of V
        context = torch.matmul(attn, V)  # [batch_size, num_heads, candidate_num, d_v]
        
        return context, attn

# Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0, "d_model must be divisible by num_attention_heads"
        
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.fc = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K=None, V=None, attn_mask=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        batch_size = Q.size(0)

        # Apply linear projections and reshape to [batch_size, num_heads, candidate_num, d_k/d_v]
        Q = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        context, attn = self.attention(Q, K, V, attn_mask)
        
        # Concatenate heads and apply final linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        output = self.fc(context)  # [batch_size, candidate_num, d_model]
        
        return output



class FACD(nn.Module):
    def __init__(self, config):
        super(FACD, self).__init__()
        self.config = config
        self.total_graph_feature = []

        self.encoder_GNN = GNNEncoder(config['out_channels'], config['out_channels']).to(config['device'])

        self.attn = AdditiveAttention(config['out_channels'], config['out_channels']).to(config['device'])
        self.g_attn_1 = nn.Linear(2 * config['out_channels'], 1)
        self.g_attn_2 = nn.Linear(2 * config['out_channels'], 1)
        self.g_attn = nn.Linear(2 * config['out_channels'], 1)
        self.t_attn = nn.Linear(2 * config['out_channels'], 1)
        
        self.encoder_student = nn.Embedding(config['stu_num'], config['out_channels'])

        self.encoder_exercise = nn.Embedding(config['prob_num'], config['out_channels'])

        self.encoder_knowledge = nn.Embedding(config['know_num'], config['out_channels'])

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


        self.time_encoder = nn.GRU(config['out_channels'], config['out_channels'], config['num_layers'], batch_first=False)
        self.self_attn = MultiHeadSelfAttention(config['out_channels'], 8)
        self.transformer = nn.TransformerEncoderLayer(d_model=config['out_channels'], nhead=8, dim_feedforward=config['out_channels'], dropout=0.1)

        self.decoder = create_dncoder(config)

    def get_graph_feature(self, edge_index, final_x):
        self.graph_feature = self.encoder_GNN.convolution(edge_index.to(torch.float32), final_x)
        return self.graph_feature
    
    def encode_time_feature(self, input):
        h = torch.zeros(self.config['num_layers'], self.config['stu_num'], self.config['out_channels']).to(self.config['device'])
        _, h = self.time_encoder(input, h)
        outputs = self.self_attn(h)
        padding = torch.zeros((self.config['prob_num'] + self.config['know_num'], self.config['out_channels'])).to(self.config['device'])
        outputs = torch.cat((torch.mean(outputs.detach(), dim = 0), padding), dim=0)
        return outputs
        
    def forward(self, student_id, exercise_id, knowledge_point, edge_index, time_input):
        student_factor = self.encoder_student.weight
        exercise_factor = self.encoder_exercise.weight
        knowledge_factor = self.encoder_knowledge.weight
        final_x = torch.cat([student_factor, exercise_factor, knowledge_factor], dim=0)

        g_feature_1 = self.get_graph_feature(edge_index, final_x)
        all_graph_feature = torch.stack(self.total_graph_feature + [g_feature_1])
        g_feature_2 = self.attn(all_graph_feature.transpose(0,1))
        concat_feature_1 = torch.cat([g_feature_1, g_feature_2], dim=1)
        g_score1 = self.g_attn_1(concat_feature_1)
        g_score2 = self.g_attn_2(concat_feature_1)            
        score = F.softmax(torch.cat([g_score1, g_score2], dim=1), dim=1)
        g_feature = score[:, 0].unsqueeze(1) * g_feature_1 + score[:, 1].unsqueeze(1) * g_feature_2

        t_feature = self.encode_time_feature(time_input)
        concat_feature_2 = torch.cat([g_feature, t_feature], dim=1)
        g_score = self.g_attn(concat_feature_2)
        t_score = self.t_attn(concat_feature_2)
        score = F.softmax(torch.cat([g_score, t_score], dim=1), dim=1)
        output = final_x + score[:, 0].unsqueeze(1) * g_feature + score[:, 1].unsqueeze(1) * t_feature
        
        return self.decoder.forward(output, student_id, exercise_id, knowledge_point)
    
    def monotonicity(self):
        self.decoder.monotonicity()
    

class FACDModel(AbstractModel):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.student_dict = {}
        self.model = None
    
    @property
    def name(self):
        return 'Dynamic Grpah Cognitive Diagnosis'
    
    def init_model(self, data):
        policy_lr=0.0005
        self.model = FACD(self.config)
        self.policy = StraightThrough(data.num_questions, data.num_questions, policy_lr, self.config)
        self.n_q = data.num_questions
        self.model.to(self.config['device'])

    def train(self, train_data):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for log in self.config['pre_train_triplets']:
            if self.student_dict.get(log[0]) is None:
                self.student_dict[log[0]] = [log[1]]
            else:
                self.student_dict[log[0]].append(log[1])
        
        seq_len = len(max(self.student_dict.values(), key=len))
        time_input = torch.zeros((seq_len, self.config['stu_num'], self.config['out_channels']))
        for stu_idx, (stu_id, exercise_ids) in enumerate(self.student_dict.items()):
            exercise_vecs = self.model.encoder_exercise(torch.tensor(exercise_ids).to(self.config['device']))
            if exercise_vecs.size(0) < seq_len:
                padding = torch.zeros((seq_len - exercise_vecs.size(0), self.config['out_channels'])).to(self.config['device'])
                exercise_vecs = torch.cat((exercise_vecs, padding), dim=0)
            time_input[:, stu_id, :] = exercise_vecs

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_step = 1
            epoch_losses = []
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(tqdm(train_loader, "Epoch %s" % ep)):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                graph = self.config['graph'].to(device)
                pred = self.model(student_ids, question_ids, concepts_emb, graph, time_input.to(self.config['device']))
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.monotonicity()
                loss += bz_loss.data.float()
                epoch_losses.append(loss.item())
                # if cnt % log_step == 0:
                #     logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
            # print(f'[{ep:03d}/{epochs}] | Loss: {np.mean(epoch_losses):.4f}, auc: {roc_auc_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())}')
    
    def l2_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.sum(torch.pow(w, 2)) / w.shape[0]
        return 0.5 * loss

    def l1_loss(self, *weights):
        loss = 0.0
        for w in weights:
            loss += torch.sum(torch.abs(w)) / w.shape[0]
        return loss

    def _loss_function(self, pred, real):
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()

    def adaptest_save(self, path):
        """
        Save the model. Only save the parameters of questions(alpha, beta)
        """
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])
    
    def adaptest_update(self, adaptest_data, sid=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True,ssid=sid)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for log in tested_dataset:
            if self.student_dict.get(log[0]) is None:
                self.student_dict[log[0]] = [log[1]]
            else:
                self.student_dict[log[0]].append(log[1])

        seq_len = len(max(self.student_dict.values(), key=len))
        time_input = torch.zeros((seq_len, self.config['stu_num'], self.config['out_channels']))
        for stu_idx, (stu_id, exercise_ids) in enumerate(self.student_dict.items()):
            exercise_vecs = self.model.encoder_exercise(torch.tensor(exercise_ids).to(self.config['device']))
            if exercise_vecs.size(0) < seq_len:
                padding = torch.zeros((seq_len - exercise_vecs.size(0), self.config['out_channels'])).to(self.config['device'])
                exercise_vecs = torch.cat((exercise_vecs, padding), dim=0)
            time_input[:, stu_id, :] = exercise_vecs
        

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_steps = 100
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb, adaptest_data.graph.to(device), time_input.to(device))
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.monotonicity()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))
        if self.config['cdm'] == 'graph':
            self.model.total_graph_feature.append(self.model.graph_feature.detach())
        return loss

    def evaluate(self, adaptest_data):
        data = adaptest_data.data
        # data = adaptest_data.get_meta_dataset()
        concept_map = adaptest_data.concept_map
        device = self.config['device']
        
        seq_len = len(max(self.student_dict.values(), key=len))
        time_input = torch.zeros((seq_len, self.config['stu_num'], self.config['out_channels']))
        for stu_idx, (stu_id, exercise_ids) in enumerate(self.student_dict.items()):
            exercise_vecs = self.model.encoder_exercise(torch.tensor(exercise_ids).to(self.config['device']))
            if exercise_vecs.size(0) < seq_len:
                padding = torch.zeros((seq_len - exercise_vecs.size(0), self.config['out_channels'])).to(self.config['device'])
                exercise_vecs = torch.cat((exercise_vecs, padding), dim=0)
            time_input[:, stu_id, :] = exercise_vecs

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs, adaptest_data.graph.to(device), time_input.to(device)).view(-1)
                pred += output.tolist()
            self.model.train()

        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)

        real = np.array(real)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        rmse = np.sqrt(mean_squared_error(real, pred))

        
        # Calculate accuracy
        threshold = 0.5  # You may adjust the threshold based on your use case
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)
        f1 = f1_score(real, binary_pred)

        return {
            'auc': auc,
            'acc': acc
        }
    
    def get_pred(self, adaptest_data):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        if len(self.student_dict) == 0:
            time_input = torch.zeros((1, self.config['stu_num'], self.config['out_channels']))
        else:
            seq_len = len(max(self.student_dict.values(), key=len))
            time_input = torch.zeros((seq_len, self.config['stu_num'], self.config['out_channels']))
            for stu_idx, (stu_id, exercise_ids) in enumerate(self.student_dict.items()):
                exercise_vecs = self.model.encoder_exercise(torch.tensor(exercise_ids).to(self.config['device']))
                if exercise_vecs.size(0) < seq_len:
                    padding = torch.zeros((seq_len - exercise_vecs.size(0), self.config['out_channels'])).to(self.config['device'])
                    exercise_vecs = torch.cat((exercise_vecs, padding), dim=0)
                time_input[:, stu_id, :] = exercise_vecs

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs, adaptest_data.graph.to(device), time_input.to(device)).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all
    
    def expected_model_change(self, sid: int, qid: int, adaptest_data, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.encoder_student.parameters(), lr=lr)

        if len(self.student_dict) == 0:
            time_input = torch.zeros((1, self.config['stu_num'], self.config['out_channels']))
        else:
            seq_len = len(max(self.student_dict.values(), key=len))
            time_input = torch.zeros((seq_len, self.config['stu_num'], self.config['out_channels']))
            for stu_idx, (stu_id, exercise_ids) in enumerate(self.student_dict.items()):
                exercise_vecs = self.model.encoder_exercise(torch.tensor(exercise_ids).to(self.config['device']))
                if exercise_vecs.size(0) < seq_len:
                    padding = torch.zeros((seq_len - exercise_vecs.size(0), self.config['out_channels'])).to(self.config['device'])
                    exercise_vecs = torch.cat((exercise_vecs, padding), dim=0)
                time_input[:, stu_id, :] = exercise_vecs

        original_weights = self.model.encoder_student.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb, adaptest_data.graph.to(device), time_input.to(device))
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.encoder_student.weight.data.clone()
        self.model.encoder_student.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb, adaptest_data.graph.to(device), time_input.to(device))
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.encoder_student.weight.data.clone()
        self.model.encoder_student.weight.data.copy_(original_weights)

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
    
    def get_BE_weights(self, pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true={}
        Pre_false={}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix={}
        for i ,_ in pred_all.items():
            w_ij_matrix[i] = {}
            for j,_ in pred_all.items(): 
                w_ij_matrix[i][j] = 0
        for i,_ in pred_all.items():
            for j,_ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11=torch.tensor(Pre_true[i],requires_grad=True)
                tensor_12=torch.tensor(Pre_true[j],requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i]*Pre_false[j]*diff_norm_00 + Pre_false[i]*Pre_true[j]*diff_norm_01 +Pre_true[i]*Pre_false[j]*diff_norm_10 + Pre_true[i]*Pre_true[j]*diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix
    
    def F_s_func(self,S_set,w_ij_matrix):
        res = 0.0
        for w_i in w_ij_matrix:
            if(w_i not in S_set):
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res +=mx
                
        return res
    
    def delta_q_S_t(self, question_id, pred_all,S_set,sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """     
        
        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}
        
        w_ij_matrix = self.get_BE_weights(sampled_dict)
        
        F_s = self.F_s_func(Sp_set,w_ij_matrix)
        
        Sp_set.append(question_id)
        F_sp =self.F_s_func(Sp_set,w_ij_matrix)
        return F_sp - F_s
    
    def bobcat_policy(self,S_set,untested_questions):
        """ get expected model change
        Args:
            S_set:list , the questions have been chosen
            untested_questions: dict, untested_questions
        Returns:
            float, expected model change
        """
        device = self.config['device']
        action_mask = [0.0] * self.n_q
        train_mask=[-0.0]*self.n_q
        for index in untested_questions:
            action_mask[index] = 1.0
        for state in S_set:
            keys = list(state.keys())
            key = keys[0]
            values = list(state.values())
            val = values[0]
            train_mask[key] = (float(val)-0.5)*2 
        action_mask = torch.tensor(action_mask).to(device)
        train_mask = torch.tensor(train_mask).to(device)
        _, action = self.policy.policy(train_mask, action_mask)
        return action.item()
