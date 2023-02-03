from torch.nn import Module
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class MLP(Module):
    def __init__(self, dim, activation, alpha=0.2):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(dim, 2 * dim)
        self.linear_2 = nn.Linear(2 * dim, dim)

        if activation == "selu":
            self.activation = nn.SELU()
        else:
            self.activation = nn.LeakyReLU(alpha)

    def forward(self, h):
        x = self.activation(self.linear_1(h))
        x = self.activation(self.linear_2(x))

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Soft_Att(Module):
    def __init__(self, dim):
        super(Soft_Att, self).__init__()
        self.dim = dim
        self.l1 = nn.Linear(dim, 1)

    def forward(self, h, h_traj, mask):
        mask = mask.unsqueeze(2)
        h = h.unsqueeze(1).repeat(1, h_traj.size(1), 1)
        # calculate weight
        beta = self.l1(h * h_traj)
        beta[mask] = float('-inf')
        beta = torch.softmax(beta, dim=1)
        z = torch.sum(beta * h_traj, 1)

        return z

class preference_learning(Module):
    def __init__(self, dim, filter_size, max_traj_len, max_time, max_location, dropout, device):
        super(preference_learning, self).__init__()
        self.dim = dim
        self.filter_size = filter_size
        self.max_traj_len = max_traj_len
        self.max_time = max_time
        self.max_location = max_location
        self.device = device

        # pos embedding
        self.pos_emb = PositionalEmbedding(self.dim)

        # cls
        self.CLS = Variable(torch.randn([self.dim], device=device), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.dim)
        self.CLS.data.uniform_(-stdv, stdv)

        # CNN
        self.activation = nn.SELU()
        self.cnn = nn.Conv1d(self.dim, self.dim, kernel_size=filter_size, stride=filter_size)

        # tf
        tf_encoder = nn.TransformerEncoderLayer(self.dim, 4, self.dim * 2, dropout=dropout, batch_first=True)
        self.tf_encoder = nn.TransformerEncoder(tf_encoder, 2)

    def getTimeEncoding(self, times):
        # Compute the positional encodings once in log space.
        pe = torch.zeros([times.size(0), times.size(1), self.dim], device=self.device).float()
        div_term = (torch.arange(0, self.dim, 2, device=self.device).float() * -(math.log(self.max_time) / self.dim)).exp()

        times = times.unsqueeze(2)
        div_term = div_term.reshape(1, 1, -1)
        pe[:, :, 0::2] = torch.sin(times * div_term)
        pe[:, :, 1::2] = torch.cos(times * div_term)

        return pe

    def getLocationEncoding(self, locations):
        # Compute the positional encodings once in log space.
        pe = torch.zeros([locations.size(0), locations.size(1), self.dim], device=self.device).float()
        div_term = (torch.arange(0, self.dim, 2, device=self.device).float() * -(math.log(self.max_location) / self.dim)).exp()

        times = locations.unsqueeze(2)
        div_term = div_term.reshape(1, 1, -1)
        pe[:, :, 0::2] = torch.sin(times * div_term)
        pe[:, :, 1::2] = torch.cos(times * div_term)

        return pe

    def forward(self, h_traj, traj_times, traj_locations, patch_mask):
        cls_mask = torch.zeros([patch_mask.size(0), 1], device=self.device).bool()
        patch_mask = torch.cat([cls_mask, patch_mask], dim=1)

        h_traj = h_traj.reshape(-1, self.max_traj_len, self.dim)
        traj_times = traj_times.reshape(-1, self.max_traj_len)
        traj_locations = traj_locations.reshape(-1, self.max_traj_len)

        h_traj += self.getTimeEncoding(traj_times) + self.getLocationEncoding(traj_locations)
        h_traj = self.activation(self.cnn(h_traj.transpose(1, 2))).transpose(1, 2)
        cls = self.CLS.reshape(1, 1, self.dim).repeat(h_traj.size(0), 1, 1)
        h_traj = torch.cat([cls, h_traj], dim=1)
        h_traj += self.pos_emb(h_traj)
        h_traj_road = self.tf_encoder(h_traj, src_key_padding_mask=patch_mask)[:, 0]

        return h_traj_road

class behavior_prediction(Module):
    def __init__(self, dim, feature_dim, car_type_num, head_num, dropout):
        super(behavior_prediction, self).__init__()
        self.dim = dim
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.encoder = nn.GRU(self.dim, self.dim, batch_first=True, dropout=dropout)
        self.multihead = nn.MultiheadAttention(self.dim, head_num, batch_first=True)
        self.mlp = MLP(self.dim, 'selu')
        self.fc = nn.Linear(self.dim, self.feature_dim)
        self.type_embedding = nn.Embedding(car_type_num, self.dim)

    def forward(self, h_road, h_hist, mask, car_types):
        h_car_types = self.type_embedding(car_types).unsqueeze(1).repeat(1, h_road.size(1), 1)
        h_road += h_car_types
        h_road, _ = self.encoder(h_road)

        # multi head
        h_pred, weight = self.multihead(h_road, h_hist, h_hist, key_padding_mask=mask)
        tokens = self.fc(self.mlp(h_pred))

        return tokens
