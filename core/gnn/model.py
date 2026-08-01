import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, global_mean_pool

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_edge_features, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Initial node feature transformation
        self.node_transform = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate)
        )

        # Core GNN layers to learn node embeddings
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=2, edge_dim=num_edge_features)
        self.norm1 = LayerNorm(hidden_channels * 2)

        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, edge_dim=num_edge_features)
        self.norm2 = LayerNorm(hidden_channels * 2)

        self.conv3 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=True, edge_dim=num_edge_features)

        # HEAD: Point of Failure Predictor (per node)
        self.pof_classifier = nn.Linear(hidden_channels, 1)
        # HEAD: Predict if POF exists (graph-level)
        self.has_pof_classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # Transform node features before GNN layers
        x = self.node_transform(x)

        # Skip GNN layers if edge_index is empty
        if edge_index.size(1) == 0:
            node_embeddings = x  # Use transformed node features directly
        else:
            # Compute node embeddings
            x = self.conv1(x, edge_index, edge_attr)
            x = self.norm1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            x = self.conv2(x, edge_index, edge_attr)
            x = self.norm2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            x = self.conv3(x, edge_index, edge_attr)
            node_embeddings = x  # Shape: [num_nodes, hidden_channels]

        # Per-node POF prediction
        pof_pred = self.pof_classifier(node_embeddings).squeeze(-1)  # Shape: [num_nodes]

        # Graph-level "has POF" prediction
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)  # Shape: [num_graphs, hidden_channels]
        else:
            # Treat single graph as batch of size 1
            graph_embedding = global_mean_pool(node_embeddings, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        has_pof_pred = self.has_pof_classifier(graph_embedding).squeeze(-1)  # Shape: [num_graphs] or scalar

        return pof_pred, has_pof_pred