import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def node2unit_pool(x, node_to_unit):
    """
    Node to Unit(Squad) Pooling
    """

    node_to_unit = node_to_unit / node_to_unit.sum(dim = 1, keepdim = True)
    x = node_to_unit.permute(0, 2, 1) @ x
    return x

def temp_pool(x):
    """
    Temporal Pooling with Mean Aggregation
    """

    x = x.mean(dim = 2)        
    return x

def masking(x, node2unit_mat, mask_prob, mask_type):
    """
    Masking function
    """

    if mask_type == 'node': # node == entity
        """
        mask node randomly
        """
        x_mask = torch.rand(x.size(0), x.size(1), x.size(2), 1) > mask_prob
        x_mask = x_mask > mask_prob
        x_mask = x_mask.repeat(1, 1, 1, x.size(3))
        x = x * x_mask

    elif mask_type == 'unit': # unit == squad
        """
        mask unit randomly
        """
        unit2node_mat = node2unit_mat.permute(0, 2, 1)
        unit_mask = torch.rand(node2unit_mat.size(0), x.size(1), node2unit_mat.size(2)) > mask_prob
        node_mask = torch.matmul(unit_mask.float(), unit2node_mat.float()).unsqueeze(3)
        node_mask = node_mask.repeat(1, 1, 1, x.size(3))

        x = x * node_mask

    elif mask_type == 'feat':
        """
        mask feature randomly
        """
        node_mask = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3)) > mask_prob
        node_mask = node_mask > mask_prob
        x = x * node_mask

    elif mask_type == 'time':
        """
        mask time randomly
        """
        time_mask = torch.rand(x.size(0), x.size(1), 1, 1) > mask_prob
        time_mask = time_mask > mask_prob
        time_mask = time_mask.repeat(1, 1, x.size(2), x.size(3))
        x = x * time_mask

    else:
        raise NotImplementedError

    return x

def add_noise(x, noise_level):

    """
    Noising Function (No noise added to force identifier, since such noise is unlikely)
    """
    # x[:,:,:,0:3] = 'PositionLat(deg)', 'PositionLon(deg)', 'PositionAlt(m)'
    # x[:,:,:,3:6] ='AttitudeYaw(deg)', 'AttitudePitch(deg)', 'AttitudeRoll(deg)'
    # x[:,:,:,6] = Speed (km/\h)
    # x[:,:,:,7] = Force Identifier
    # x[:,:,:,8:] = 'T_Road', 'T_Forest', 'T_OpenLane', 'T_HidingPlace', 'T_Building'
    assert noise_level in [0.8, 1.6, 2.4, 3.2, 4.0]
    noise_prob = noise_level * 0.1

    # add gaussian noise to continuous features
    gaussian_noise = torch.distributions.normal.Normal(0, noise_level)
    noise = gaussian_noise.sample(x[:,:,:,:7].shape)
    x[:,:,:,:7] = x[:,:,:,:7] + noise
    
    noise_idx = torch.rand(x[:,:,:,8:].shape) > noise_prob
    x[:,:,:,8:] = torch.where(noise_idx, x[:,:,:,8:], 1-x[:,:,:,8:])

    return x

    
class PositionalEncoding(torch.nn.Module):

    def __init__(
        self,
        out_channels: int,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
    ):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with "
                             f"odd 'out_channels' (got {out_channels}).")

        self.out_channels = out_channels
        self.base_freq = base_freq
        self.granularity = granularity

        frequency = torch.logspace(0, 1, out_channels // 2, base_freq)
        self.register_buffer('frequency', frequency)

        self.reset_parameters()

    def reset_parameters(self):
        pass


    def forward(self, x: Tensor) -> Tensor:
        x = x / self.granularity if self.granularity != 1.0 else x
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'
