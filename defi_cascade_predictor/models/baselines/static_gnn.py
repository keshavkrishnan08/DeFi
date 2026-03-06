"""
Static GNN baseline: GAT without temporal components.
Uses the same graph structure but without memory or time encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class StaticGNNCascadePredictor(nn.Module):
    """Graph Attention Network baseline without temporal components.

    Ablation: measures the contribution of temporal modeling by
    comparing against a static graph snapshot approach.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prediction_horizons = prediction_horizons

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Final projection
        self.final_proj = nn.Linear(hidden_dim * heads, hidden_dim)

        # Graph pooling attention
        self.pool_attn = nn.Linear(hidden_dim, 1)

        # Prediction heads
        self.prediction_heads = nn.ModuleDict()
        for h in prediction_horizons:
            self.prediction_heads[f"head_{h}h"] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            node_features: [num_nodes, feature_dim].
            edge_index: [2, num_edges] (homogeneous).

        Returns:
            Dict of predictions.
        """
        x = self.input_proj(node_features)

        for gat, norm in zip(self.gat_layers, self.norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = F.elu(x)

        x = self.final_proj(x)  # [num_nodes, hidden_dim]

        # Attention pooling
        attn = F.softmax(self.pool_attn(x), dim=0)
        graph_embed = (attn * x).sum(dim=0, keepdim=True)
        max_embed = x.max(dim=0, keepdim=True).values
        combined = torch.cat([graph_embed, max_embed], dim=-1)

        outputs = {}
        for h in self.prediction_horizons:
            outputs[f"cascade_{h}h"] = self.prediction_heads[f"head_{h}h"](
                combined
            ).squeeze()

        outputs["severity"] = self.severity_head(combined).squeeze()
        return outputs
