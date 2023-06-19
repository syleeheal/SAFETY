import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PositionalEncoding, node2unit_pool, temp_pool

"""
-------------------Prediction Models-------------------
"""


class Spatio_Temporal_Att(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        t_edge_index = torch.tensor([[i, j] for i in range(self.args.time_steps) for j in range(self.args.time_steps)], dtype = torch.long).t().contiguous()
        self.t_edge_index = t_edge_index.to(self.args.device)
        
        self.pos_dim = 4 
        PE = PositionalEncoding(out_channels = self.pos_dim)
        self.pos = PE(torch.arange(0, self.args.time_steps).unsqueeze(1)).to(self.args.device) 
    
        self.layer_init()
    
    def layer_init(self):

        in_channels = self.args.num_node_feat + self.pos_dim
        hid_channels = self.args.hid_dim * self.args.num_heads
        out_channels = self.args.num_unit_labels
        
        self.att_block = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.att_block.append(GNN(self.args, in_channels, self.args.hid_dim, self.args.num_heads))
            in_channels = hid_channels
            self.att_block.append(GNN(self.args, in_channels, self.args.hid_dim, self.args.num_heads))

        self.classifiers = Classifiers(self.args, hid_channels, out_channels)
        self.dropout = nn.Dropout(self.args.dropout)
          
    def forward(self, x, edge_index, node_to_unit):

        pe = self.pos.repeat(x.size(0), x.size(2), 1, 1).permute(0, 2, 1, 3)
        x = torch.cat([x, pe], dim = -1)

        
        for i in range(self.args.num_layers):
            x = self.att_block[i](x, edge_index)
            
        x = x.permute(0, 2, 1, 3)
        
        for j in range(self.args.num_layers):
            k = i + j + 1
            x = self.att_block[k](x, self.t_edge_index)
        
        x = temp_pool(x)
        x = node2unit_pool(x, node_to_unit)
        task_prob, atk_prob = self.classifiers(x)
        
        return task_prob, atk_prob

class RNN_Classifier(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        in_channels = self.args.num_node_feat
        hid_channels = self.args.hid_dim
        out_channels = self.args.num_unit_labels

        self.rnn = RNN(self.args, in_channels, hid_channels)
        self.classifiers = Classifiers(self.args, hid_channels, out_channels)
    
    def forward(self, x, edge_index, node_to_unit):

        x = self.rnn(x)
        x = node2unit_pool(x, node_to_unit)
        task_prob, atk_prob = self.classifiers(x)
        
        return task_prob, atk_prob

class GRU_Classifier(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        in_channels = self.args.num_node_feat
        hid_channels = self.args.hid_dim
        out_channels = self.args.num_unit_labels

        self.gru = GRU(self.args, in_channels, hid_channels)
        self.classifiers = Classifiers(self.args, hid_channels, out_channels)
    
    def forward(self, x, edge_index, node_to_unit):

        x = self.gru(x)
        x = node2unit_pool(x, node_to_unit)
        task_prob, atk_prob = self.classifiers(x)
        
        return task_prob, atk_prob

class MLP_Classifier(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        in_channels = self.args.num_node_feat * self.args.time_steps
        hid_channels = self.args.hid_dim
        out_channels = self.args.num_unit_labels
        self.mlp = MLP(self.args, in_channels, hid_channels)
        self.classifiers = Classifiers(self.args, hid_channels, out_channels)
    
    def forward(self, x, edge_index, node_to_unit):

        x = x.reshape(x.size(0), x.size(2), -1) # concat time
        x = node2unit_pool(x, node_to_unit)
        x = self.mlp(x)
        task_prob, atk_prob = self.classifiers(x)
        
        return task_prob, atk_prob

class Panoramic_LSTM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        in_channels = self.args.num_node_feat
        hid_channels = self.args.hid_dim
        out_channels = self.args.num_unit_labels

        self.cnn = CNN(self.args, in_channels, hid_channels)
        self.lstm = RNN(self.args, in_channels, hid_channels)
        self.classifiers = Classifiers(self.args, hid_channels, out_channels)
    
    def forward(self, x, edge_index, node_to_unit):

        x = self.cnn(x)
        x = self.lstm(x)
        x = node2unit_pool(x, node_to_unit)
        task_prob, atk_prob = self.classifiers(x)
        
        return task_prob, atk_prob



"""
-------------------Modules-------------------
"""

class GNN(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels, num_heads):
        super().__init__()

        self.args = args
        self.att_conv = Attention_Conv(in_channels = in_channels, 
                            out_channels = out_channels, 
                            heads = num_heads,
                            dropout = self.args.dropout
                            )

        self.dropout = nn.Dropout(p = self.args.dropout)
        hid_dim = self.args.hid_dim * self.args.num_heads

        self.act = nn.ELU()
        self.ln = nn.LayerNorm(hid_dim)
        
    def forward(self, x, edge_index):
        x = self.dropout(x)
        h = self.att_conv(x, edge_index)
        h = self.ln(h)
        h = self.act(h)
        
        return h ### size = (time_steps, num_nodes, hidden_dim)

class RNN(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args

        if self.args.num_layers == 1:
            dropout = 0
        else:
            dropout = self.args.dropout

        self.rnn = nn.LSTM(input_size = in_channels,
                            hidden_size = out_channels,
                            num_layers = self.args.num_layers,
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

        if self.args.num_layers == 1:
            dropout = 0
        else:
            dropout = self.args.dropout

        
        self.rnn = nn.GRU(input_size = in_channels,
                        hidden_size = int(out_channels/2),
                        num_layers = self.args.num_layers,
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
        self.out_channels = int(self.args.time_steps)

        self.conv = nn.Conv2d(in_channels = self.args.time_steps,
                            out_channels = self.args.time_steps,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.init_layers()

    def init_layers(self):
        self.mlp_layers = nn.ModuleList()

        in_channels = self.in_channels

        for i in range(self.args.num_layers-1):

            self.mlp_layers.append(nn.Sequential(
                nn.Dropout(p = self.args.dropout),
                nn.Linear(in_channels, self.out_channels),
                nn.LayerNorm(self.out_channels),
                nn.ELU()
            ))

            in_channels = self.out_channels


    def forward(self, x):

        for i in range(int(self.args.num_layers-1)):
            x = self.mlp_layers[i](x)

        return x

class Classifiers(torch.nn.Module):

    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels

        unit_adj_matrix = torch.ones(self.args.num_units, self.args.num_units) 
        self.unit_adj_idx = unit_adj_matrix.nonzero().t() 
        
        self.init_layers()

    def init_layers(self):
        self.task_pred = nn.Sequential(
            nn.Dropout(p = self.args.dropout),
            nn.Linear(self.in_channels, self.out_channels),
            nn.Softmax(dim = -1)
        )

        self.attack_pred = nn.Sequential(
            nn.Dropout(p = self.args.dropout),
            nn.Linear(self.in_channels, 1),
            nn.Sigmoid()
        )

        if self.args.pred == 'attack': self.task_pred = None
        if self.args.pred == 'task': self.attack_pred = None

    def forward(self, x):

        if self.args.pred == 'attack':
            edge_x = x[:, self.unit_adj_idx[0], :] * x[:, self.unit_adj_idx[1], :]
            attack_prob = self.attack_pred(edge_x)
            attack_prob = attack_prob.view(attack_prob.size(0), self.args.num_units, self.args.num_units)
            
            task_prob = None


        elif self.args.pred == 'task':
            task_prob = self.task_pred(x)

            attack_prob = None


        elif self.args.pred == 'joint':
            task_prob = self.task_pred(x)

            edge_x = x[:, self.unit_adj_idx[0], :] * x[:, self.unit_adj_idx[1], :]
            attack_prob = self.attack_pred(edge_x)
            attack_prob = attack_prob.view(attack_prob.size(0), self.args.num_units, self.args.num_units)

        return task_prob, attack_prob


"""
-------------------Attention Layer-------------------
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax


class Attention_Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(Attention_Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)
        self.lin_skip = Linear(in_channels, heads * out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        B = x.size(0) # sample size
        T = x.size(1) # num windows
        N = x.size(2) # num nodes

        x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(B, T, N, H, C)
        key = self.lin_key(x[0]).view(B, T, N, H, C)
        value = self.lin_value(x[0]).view(B, T, N, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value, size=None)
        out = out.view(B, T, N, H * C)

        x_r = self.lin_skip(x[1])
        out = out + x_r

        return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j

        out = out * alpha.unsqueeze(-1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


