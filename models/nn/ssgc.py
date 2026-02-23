import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SSGConv, global_mean_pool


class SSGC(nn.Module):
    """
    SSGC backbone for graph classification.
    """

    def __init__(self, num_classes=3, embedding_dim=128):
        super().__init__()
        hidden = 256
        K1 = 2
        K2 = 4
        alpha = 0.1

        self.proj = nn.LazyLinear(hidden)

        self.conv1 = SSGConv(in_channels=hidden, out_channels=hidden, K=K1, alpha=alpha)
        self.conv2 = SSGConv(in_channels=hidden, out_channels=hidden, K=K2, alpha=alpha)

        self.bn = nn.BatchNorm1d(hidden * 2)
        self.dropout = 0.2

        self.emb_dim_fix_layer = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.proj(x)  # [N, hidden]

        x1 = F.relu(self.conv1(x, edge_index))  # [N, hidden]
        x2 = F.relu(self.conv2(x, edge_index))  # [N, hidden]
        x_cat = torch.cat([x1, x2], dim=-1)  # [N, 2*hidden]
        x_cat = self.bn(x_cat)
        x_cat = F.dropout(x_cat, p=self.dropout, training=self.training)

        g = global_mean_pool(x_cat, batch)  # [B, 2*hidden]
        embedding = self.emb_dim_fix_layer(g)  # [B, embedding_dim]
        out = self.classifier(embedding)
        return embedding, out
