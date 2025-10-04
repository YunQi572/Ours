import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from .GCN import *

def load_model(name, input_dim, hidden_dim, output_dim, num_layers = 2, dropout = 0.5):
    # if name == "GCN":
    #     return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout)
    return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout)