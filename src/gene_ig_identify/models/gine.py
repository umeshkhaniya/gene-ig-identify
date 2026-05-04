"""Shared GINE model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool


class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim, num_classes, num_layers, dropout_rate):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINEConv(Sequential(Linear(in_channels, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Dropout(dropout_rate), Linear(hidden_dim, hidden_dim)), edge_dim=edge_in_channels))
        for _ in range(1, num_layers):
            self.convs.append(GINEConv(Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Dropout(dropout_rate), Linear(hidden_dim, hidden_dim)), edge_dim=edge_in_channels))
        pool_output_dim = hidden_dim * 3 * num_layers
        self.lin1 = Linear(pool_output_dim, pool_output_dim)
        self.lin2 = Linear(pool_output_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        pooled_outputs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            pooled_outputs.append(global_mean_pool(x, batch))
            pooled_outputs.append(global_add_pool(x, batch))
            pooled_outputs.append(global_max_pool(x, batch))
        h = torch.cat(pooled_outputs, dim=1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)

