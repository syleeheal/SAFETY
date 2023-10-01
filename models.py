import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

from typing import Optional, Tuple, Union

from utils import PositionalEncoding, entity2squad_pool, temp_pool

"""
Models
"""

class Spatio_Temporal_Att(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):

        super().__init__()

        self.args = args

        # spatial att index init
        s_att_index = torch.tensor([[i, j] for i in range(self.args.num_entities) for j in range(self.args.num_entities)], dtype = torch.long).t().contiguous()
        self.s_att_index = s_att_index.to(self.args.device)

        # temporal att index init
        t_att_index = torch.tensor([[i, j] for i in range(self.args.t_sample_size) for j in range(self.args.t_sample_size)], dtype = torch.long).t().contiguous()
        self.t_att_index = t_att_index.to(self.args.device)
        
        # positional encoder init
        self.pos_dim = 4 
        PE = PositionalEncoding(out_channels = self.pos_dim)
        self.pos = PE(torch.arange(0, self.args.t_sample_size).unsqueeze(1)).to(self.args.device) 
        in_channels = in_channels + self.pos_dim
    
        # model init
        self.layer_init(in_channels, out_channels)
    
    def layer_init(self, in_channels, out_channels):
        
        # S/T-ATT layer init
        self.att_nets = nn.ModuleList()
        self.att_nets.append(Att_Net(self.args, in_channels, self.args.hid_dim))
        self.att_nets.append(Att_Net(self.args, self.args.hid_dim, self.args.hid_dim))

        # classification layer init
        self.classifiers = Classifiers(self.args, self.args.hid_dim, out_channels)
          
    def forward(self, x, entity2squad_idx, squad2squad_idx):

        B, T, N, C = x.size()

        # position encoding
        pe = self.pos.repeat(x.size(0), x.size(2), 1, 1).permute(0, 2, 1, 3)
        x = torch.cat([x, pe], dim = -1) # size = (B, T, N, C + pos_dim)

        # spatial attention
        x = self.att_nets[0](x, self.s_att_index)
        
        # dimension permutation 
        x = x.permute(0, 2, 1, 3) # size = (B, N, T, C)
        
        # temporal attention
        x = self.att_nets[1](x, self.t_att_index)
        
        # temporal and squad aggregation
        x = temp_pool(x)
        x = entity2squad_pool(x, entity2squad_idx)

        # classification
        int_prob, atk_prob = self.classifiers(x, squad2squad_idx)
        
        return int_prob, atk_prob

class LSTM_Classifier(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.lstm = LSTM(self.args, in_channels, self.args.hid_dim)
        self.classifiers = Classifiers(self.args, self.args.hid_dim, out_channels)
    
    def forward(self, x, entity2squad_idx, squad2squad_idx):

        x = self.lstm(x)
        x = entity2squad_pool(x, entity2squad_idx)
        int_prob, atk_prob = self.classifiers(x, squad2squad_idx)
        
        return int_prob, atk_prob

class GRU_Classifier(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.gru = GRU(self.args, in_channels, self.args.hid_dim)
        self.classifiers = Classifiers(self.args, self.args.hid_dim, out_channels)
    
    def forward(self, x, entity2squad_idx, squad2squad_idx):

        x = self.gru(x)
        x = entity2squad_pool(x, entity2squad_idx)
        int_prob, atk_prob = self.classifiers(x, squad2squad_idx)
        
        return int_prob, atk_prob

class MLP_Classifier(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args

        in_channels = in_channels * self.args.t_sample_size
        self.mlp = MLP(self.args, in_channels, self.args.hid_dim)
        self.classifiers = Classifiers(self.args, self.args.hid_dim, out_channels)
    
    def forward(self, x, entity2squad_idx, squad2squad_idx):

        x = x.reshape(x.size(0), x.size(2), -1) # concat time

        x = entity2squad_pool(x, entity2squad_idx)
        x = self.mlp(x)
        int_prob, atk_prob = self.classifiers(x, squad2squad_idx)
        
        return int_prob, atk_prob

class Panoramic_LSTM(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.cnn = CNN(self.args, in_channels, self.args.hid_dim)
        self.lstm = LSTM(self.args, in_channels, self.args.hid_dim)
        self.classifiers = Classifiers(self.args, self.args.hid_dim, out_channels)
    
    def forward(self, x, entity2squad_idx, squad2squad_idx):

        x = self.cnn(x)
        x = self.lstm(x)
        x = entity2squad_pool(x, entity2squad_idx)
        int_prob, atk_prob = self.classifiers(x, squad2squad_idx)
        
        return int_prob, atk_prob



"""
Modules
"""

class Att_Net(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.att = Attention_Layer(in_channels = in_channels, 
                            out_channels = out_channels, 
                            dropout = self.args.dropout,
                            )

        self.dropout = nn.Dropout(p = self.args.dropout)

        self.act = nn.ELU()
        self.ln = nn.LayerNorm(self.args.hid_dim)
        
    def forward(self, x, att_index):
        x = self.dropout(x)
        h = self.att(x, att_index)
        h = self.ln(h)
        h = self.act(h)
        
        return h ### size = (t_sample_size, num_entities, hidden_dim)

class LSTM(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args

        if int(self.args.num_layers) == 1:
            dropout = 0
        else:
            dropout = self.args.dropout

        self.rnn = nn.LSTM(input_size = in_channels,
                            hidden_size = out_channels,
                            num_layers = int(self.args.num_layers),
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = False)

        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p = self.args.dropout)
        
    def forward(self, x):
        
        B, T, N, C = x.size()
        
        x = x.reshape(B * N, T, C)
        
        x = self.dropout(x)
        x = self.rnn(x)[0]
        
        x = x.reshape(B, T, N, -1)
        
        x = x[:,-1,:,:] 
        x = self.ln(x)

        return x

class GRU(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args

        if int(self.args.num_layers) == 1:
            dropout = 0
        else:
            dropout = self.args.dropout

        
        self.rnn = nn.GRU(input_size = in_channels,
                        hidden_size = int(out_channels/2),
                        num_layers = int(self.args.num_layers),
                        batch_first = True,
                        dropout = dropout,
                        bidirectional = True
                        )

        
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p = self.args.dropout)

        self.linear = nn.Linear(out_channels, out_channels)
        self.C = out_channels
        
    def forward(self, x):

        B, T, N, C = x.size()
        
        x = x.reshape(B * N, T, C)
        
        x = self.dropout(x)
        x = self.rnn(x)[0]
        
        x = x.reshape(B, T, N, -1)

        att = self.linear(x) 
        att = torch.tanh(att)
        att = torch.softmax(att, dim = 1)
        x = (x * att).sum(dim = 1)

        x = self.ln(x)

        return x

class CNN(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.out_channels = int(self.args.t_sample_size)

        self.conv = nn.Conv2d(in_channels = self.args.t_sample_size,
                            out_channels = self.args.t_sample_size,
                            kernel_size = (1, 3),
                            stride = (1, 1),
                            padding = (0, 1),
                            bias = True)

        self.act = nn.ELU()
        self.ln = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(p = self.args.dropout)
        
    def forward(self, x):

        B, T, N, C = x.size()
        
        x = x.reshape(B * N, T, C, 1)

        x = self.dropout(x)
        x = self.conv(x)
        x = self.act(x)
        x = x.reshape(B, self.out_channels, N, -1)
        x = self.ln(x)

        # reduce time dimension by time pooling factor of 2
        x1 = x[:,::2,:,:]
        x2 = x[:,1::2,:,:]
        x = (x1 + x2) / 2
        
        return x

class MLP(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.mlp_layers = nn.ModuleList()

        for i in range(int(self.args.num_layers)-1):

            self.mlp_layers.append(nn.Sequential(
                nn.Dropout(p = self.args.dropout),
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ELU()
            ))

            in_channels = out_channels


    def forward(self, x):

        for i in range(int(self.args.num_layers-1)):
            x = self.mlp_layers[i](x)

        return x

class Classifiers(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        
        self.int_pred = nn.Sequential(
            nn.Dropout(p = self.args.dropout),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

        self.atk_pred = nn.Sequential(
            nn.Dropout(p = self.args.dropout),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

        if self.args.pred == 'y_int': self.atk_pred = None
        if self.args.pred == 'y_atk': self.int_pred = None


    def forward(self, x, squad2squad_idx):


        if self.args.pred == 'y_int':
            int_prob = self.int_pred(x)
            atk_prob = None


        elif self.args.pred == 'y_atk':
            squad_pair_x = x[:, squad2squad_idx[0], :] * x[:, squad2squad_idx[1], :]
            atk_prob = self.atk_pred(squad_pair_x)
            atk_prob = atk_prob.view(atk_prob.size(0), squad2squad_idx.max() + 1, squad2squad_idx.max() + 1)
            int_prob = None


        return int_prob, atk_prob


"""
Attention Layer
"""



class Attention_Layer(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        dropout: float = 0.,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(Attention_Layer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.lin_key = Linear(in_channels, out_channels)
        self.lin_query = Linear(in_channels, out_channels)
        self.lin_value = Linear(in_channels, out_channels)
        self.lin_skip = Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()


    def forward(self, x, att_index):

        C = self.out_channels # hidden dim
        B = x.size(0) # batch size 
        T = x.size(1) # t_sample_size, when in spatial attention; num num_entities, when in temporal attention
        N = x.size(2) # num num_entities, when in spatial attention; t_sample_size, when in temporal attention

        x: PairTensor = (x, x)
        
        # QKV computation
        query = self.lin_query(x[1])
        key = self.lin_key(x[0])
        value = self.lin_value(x[0])
        
        # Attention Propagation
        out = self.propagate(edge_index=att_index, 
                             query=query.view(N, B, T, C), 
                             key=key.view(N, B, T, C), 
                             value=value.view(N, B, T, C),
                             ).view(B, T, N, C)

        # Skip connection
        x_r = self.lin_skip(x[1])
        out = out + x_r

        return out


    def message(self, query_i, key_j, value_j, edge_index):

        # Compute attention coefficients.
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        
        # softmax
        alpha = softmax(alpha, edge_index[0])

        # dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # attention propagation
        return value_j * alpha.unsqueeze(-1)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, 'f'{self.out_channels})')
