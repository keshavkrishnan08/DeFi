"""
Temporal attention layer for the Temporal Graph Network.
Implements multi-head attention over temporal node neighborhoods
with time-aware positional encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoding(nn.Module):
    """Learnable time encoding using Bochner's theorem-inspired approach.

    Maps time deltas to a fixed-dimensional representation using
    learnable frequency parameters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time delta tensor of shape [batch_size] or [batch_size, 1].

        Returns:
            Time encoding of shape [batch_size, dim].
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.float()
        output = torch.cos(self.w(t))
        return output


class TemporalAttentionLayer(nn.Module):
    """Multi-head temporal attention layer.

    Computes attention over temporal node neighborhoods, incorporating
    time encodings, source node features, edge features, and neighbor
    features to produce updated node embeddings.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Input projection dimension
        total_input_dim = node_dim + edge_dim + time_dim

        # Multi-head attention projections
        self.query_proj = nn.Linear(node_dim + time_dim, output_dim)
        self.key_proj = nn.Linear(total_input_dim, output_dim)
        self.value_proj = nn.Linear(total_input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        query_node_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        edge_features: torch.Tensor,
        time_encodings: torch.Tensor,
        neighbor_time_encodings: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query_node_features: [batch, node_dim] - target node features.
            neighbor_node_features: [batch, n_neighbors, node_dim].
            edge_features: [batch, n_neighbors, edge_dim].
            time_encodings: [batch, time_dim] - query node time encoding.
            neighbor_time_encodings: [batch, n_neighbors, time_dim].
            mask: [batch, n_neighbors] - boolean mask (True = ignore).

        Returns:
            Updated node embeddings [batch, output_dim].
        """
        batch_size = query_node_features.size(0)
        n_neighbors = neighbor_node_features.size(1)

        # Construct query from target node + its time encoding
        query_input = torch.cat([query_node_features, time_encodings], dim=-1)
        Q = self.query_proj(query_input)  # [batch, output_dim]

        # Construct key/value from neighbor features + edge features + time
        neighbor_input = torch.cat(
            [neighbor_node_features, edge_features, neighbor_time_encodings],
            dim=-1,
        )  # [batch, n_neighbors, total_input_dim]
        K = self.key_proj(neighbor_input)  # [batch, n_neighbors, output_dim]
        V = self.value_proj(neighbor_input)  # [batch, n_neighbors, output_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_neighbors, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, n_neighbors, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        # Q: [batch, heads, 1, head_dim]
        # K, V: [batch, heads, n_neighbors, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, heads, 1, n_neighbors]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_neighbors]
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch, heads, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, self.output_dim)

        # Output projection + residual + norm
        output = self.output_proj(context)
        output = self.dropout(output)

        # Residual connection (project query to match output dim if needed)
        if query_node_features.size(-1) != self.output_dim:
            residual = nn.functional.linear(
                query_node_features,
                torch.eye(self.output_dim, self.node_dim, device=output.device),
            )
        else:
            residual = query_node_features
        output = self.layer_norm(output + residual)

        # Feed-forward + residual
        ffn_output = self.ffn(output)
        output = self.ffn_norm(output + ffn_output)

        return output
