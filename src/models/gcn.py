import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj


class GCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
