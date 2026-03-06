"""
Message passing layer for the Temporal Graph Network.
Implements graph attention-based message passing over the
DeFi composability graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing as PyGMessagePassing
from torch_geometric.utils import softmax


class MessagePassingLayer(PyGMessagePassing):
    """Graph attention-based message passing for the composability graph.

    Extends PyG's MessagePassing with edge-type-aware attention
    and temporal node memory integration.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 16,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.Tensor(1, heads, self.head_dim))

        # Output
        if concat:
            self.lin_out = nn.Linear(out_dim, out_dim)
        else:
            self.lin_out = nn.Linear(self.head_dim, out_dim)

        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim].
            edge_index: [2, num_edges].
            edge_attr: Optional edge features [num_edges, edge_dim].

        Returns:
            Updated node features [num_nodes, out_dim].
        """
        # Linear projections
        x_src = self.lin_src(x)
        x_dst = self.lin_dst(x)

        # Reshape for multi-head attention
        x_src = x_src.view(-1, self.heads, self.head_dim)
        x_dst = x_dst.view(-1, self.heads, self.head_dim)

        # Process edge features
        if edge_attr is not None:
            if edge_attr.size(-1) != self.lin_edge.in_features:
                # Pad or project edge features
                edge_attr_proc = torch.zeros(
                    edge_attr.size(0),
                    self.lin_edge.in_features,
                    device=edge_attr.device,
                )
                min_dim = min(edge_attr.size(-1), self.lin_edge.in_features)
                edge_attr_proc[:, :min_dim] = edge_attr[:, :min_dim]
                edge_attr = edge_attr_proc
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)

        # Message passing
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            size=None,
        )

        if self.concat:
            out = out.view(-1, self.out_dim)
        else:
            out = out.mean(dim=1)

        out = self.lin_out(out)
        out = self.dropout_layer(out)

        # Residual connection
        if x.size(-1) == self.out_dim:
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out)

        return out

    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr=None,
        size_i=None,
    ) -> torch.Tensor:
        """Compute messages with attention weights."""
        # Attention scores
        alpha_src = (x_j * self.att_src).sum(dim=-1)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)

        alpha = alpha_src + alpha_dst

        if edge_attr is not None:
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted messages
        msg = x_j * alpha.unsqueeze(-1)
        return msg


class HeteroMessagePassingLayer(nn.Module):
    """Handles message passing over heterogeneous edge types.

    Applies separate MessagePassingLayers for each edge type and
    aggregates the results with a residual connection to the input.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_types: list[str],
        edge_dim: int = 16,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.out_dim = out_dim

        # One message passing layer per edge type
        self.mp_layers = nn.ModuleDict({
            etype: MessagePassingLayer(
                in_dim, out_dim, edge_dim, heads, dropout
            )
            for etype in edge_types
        })

        # Gated aggregation: learn per-edge-type importance
        self.edge_type_gates = nn.ParameterDict({
            etype: nn.Parameter(torch.ones(1))
            for etype in edge_types
        })

        # Aggregation: mean of active edge types + projection
        self.aggregate = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dims mismatch
        self.residual_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor],
        edge_attr_dict: dict[str, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim].
            edge_index_dict: Dict mapping edge_type -> [2, num_edges].
            edge_attr_dict: Optional dict mapping edge_type -> edge features.

        Returns:
            Updated node features [num_nodes, out_dim].
        """
        if edge_attr_dict is None:
            edge_attr_dict = {}

        # Weighted sum of active edge-type outputs (skip empty types)
        weighted_sum = torch.zeros(x.size(0), self.out_dim, device=x.device)
        total_weight = torch.zeros(1, device=x.device)

        for etype in self.edge_types:
            if etype in edge_index_dict and edge_index_dict[etype].size(1) > 0:
                edge_attr = edge_attr_dict.get(etype)
                out = self.mp_layers[etype](
                    x, edge_index_dict[etype], edge_attr
                )
                gate = torch.sigmoid(self.edge_type_gates[etype])
                weighted_sum = weighted_sum + gate * out
                total_weight = total_weight + gate

        # Normalize by number of active edge types
        if total_weight.item() > 0:
            weighted_sum = weighted_sum / total_weight

        output = self.aggregate(weighted_sum)
        output = self.dropout(output)

        # Residual connection
        residual = self.residual_proj(x)
        output = self.layer_norm(output + residual)

        return output
