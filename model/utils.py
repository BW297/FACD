import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super().__init__()
        # actor
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        hidden_state = self.obs_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        actions = hard_sample(logits)
        return actions

class StraightThrough:
    def __init__(self, state_dim, action_dim, lr,  config):
        self.lr = lr
        device = config['device']
        self.betas = config['betas']
        self.policy = Actor(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=self.betas)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

def create_dncoder(config):
    if config['decoder'] == 'ncd':
        return NCDDecoder(config).to(config['device'])
    elif config['decoder'] == 'irt':
        return IRTDecoder(config).to(config['device'])



class NCDDecoder(nn.Module):

    def __init__(
            self, config
    ):
        super().__init__()
        self.layers = Positive_MLP(config).to(config['device'])
        self.transfer_student_layer = nn.Linear(config['out_channels'], config['know_num']).to(config['device'])
        self.transfer_exercise_layer = nn.Linear(config['out_channels'], config['know_num']).to(config['device'])
        self.e_discrimination = nn.Embedding(config['prob_num'], 1)
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        state = (torch.sigmoid(self.transfer_student_layer(z[student_id])) - torch.sigmoid(
            self.transfer_exercise_layer(z[self.config['stu_num'] + exercise_id]))) * knowledge_point
        return self.layers.forward(state).view(-1)

    def get_mastery_level(self, z):
        return torch.sigmoid(self.transfer_student_layer(z[:self.config['stu_num']])).detach().cpu().numpy()

    def monotonicity(self):
        none_neg_clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)

class IRTDecoder(nn.Module):

    def __init__(
            self, config
    ):
        super().__init__()
        self.theta = nn.Linear(config['out_channels'], config['num_dim'])
        self.alpha = nn.Linear(config['out_channels'], config['num_dim'])
        self.beta = nn.Linear(config['out_channels'], 1)
        self.config = config
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, z, student_id, exercise_id, knowledge_point):
        theta = self.theta(z[student_id])
        alpha = self.alpha(z[self.config['stu_num'] + exercise_id])
        beta = self.beta(z[self.config['stu_num'] + exercise_id])
        pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred.view(-1)

    def get_mastery_level(self, z):
        return torch.sigmoid(z[:self.config['stu_num']]).detach().cpu().numpy()

    def monotonicity(self):
        pass


def get_mlp_encoder(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, 512),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.PReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, out_channels),
    )

def Positive_MLP(config, num_layers=3, hidden_dim=512, dropout=0.5):
    layers = []
    layers.append(nn.Linear(config['know_num'], 128))
    layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(128, 64))
    layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(64, 1))
    layers.append(nn.Sigmoid())
    layers = nn.Sequential(*layers)
    return layers