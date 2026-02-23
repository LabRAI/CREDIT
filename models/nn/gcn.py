import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """
    GCN backbone for graph classification.
    """

    def __init__(self, num_classes=3, embedding_dim=128):
        super().__init__()
        hidden = 256
        self.conv1 = GCNConv(in_channels=-1, out_channels=hidden)
        self.conv2 = GCNConv(in_channels=hidden, out_channels=hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = 0.2

        self.emb_dim_fix_layer = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # global pooling to get graph level representation
        g = global_mean_pool(x, batch)  # [B, hidden]

        embedding = self.emb_dim_fix_layer(g)  # [B, embedding_dim]
        out = self.classifier(embedding)
        return embedding, out
