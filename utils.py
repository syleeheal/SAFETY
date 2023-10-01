import torch
from torch import Tensor

def entity2squad_pool(x, entity2squad_idx):
    """
    Entity to Squad Pooling
    """

    entity2squad_idx = entity2squad_idx / entity2squad_idx.sum(dim = 1, keepdim = True)
    entity2squad_idx[entity2squad_idx != entity2squad_idx] = 0
    x = entity2squad_idx.permute(0, 2, 1) @ x
    return x

def temp_pool(x):
    """
    Temporal Pooling with Mean Aggregation
    """

    x = x.mean(dim = 2)        
    
    return x

def load_hyperparam(args):
    
    path = './SAFETY/best_hyperparam/' + args.model + '_' + args.pred + '.txt'

    # read txt file
    with open(path, 'r') as f: 
        for i in range(2): line = f.readline()
        
        line = line[23:-2]
        
        line = line.replace(',', ' ')
        num_hyperparam = int((len(line.split('\''))-1) / 2)

        for i in range(num_hyperparam):
            strings = line.split('\'')[1:][i*2]
            value = line.split('\'')[1:][(i*2)+1].split()[1]
            
            args.__dict__[strings] = float(value)
    
    return args

    
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
