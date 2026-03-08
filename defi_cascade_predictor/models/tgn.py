"""
Temporal Graph Network (TGN) for DeFi Liquidation Cascade Prediction.

Core architecture:
  1. Memory Module — per-node GRU memory tracking protocol state evolution
  2. Time Encoding — learnable continuous-time positional encoding
  3. Feature Gate — learned feature selection (TGIB-inspired)
  4. Embedding Module — graph attention over temporal neighborhoods
  5. Temporal Attention — attends over neighbor node history with time encoding
  6. Multi-Scale Temporal Context — pools memory at multiple recency scales
  7. Multi-horizon Prediction Heads — cascade probability at 1d/3d/7d/30d

Reference: Rossi et al., "Temporal Graph Networks for Deep Learning on
Dynamic Graphs", ICML 2020 Workshop on GRL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .layers.memory_module import MemoryModule
from .layers.temporal_attention import TimeEncoding, TemporalAttentionLayer
from .layers.message_passing import HeteroMessagePassingLayer


class TemporalGraphNetwork(nn.Module):
    """TGN adapted for DeFi composability graph cascade prediction.

    The model processes temporal snapshots of the DeFi composability graph,
    maintaining per-protocol memory vectors that capture evolving risk states.
    At each timestep, it produces multi-horizon cascade probability predictions.
    """

    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        edge_types: list[str],
        memory_dim: int = 128,
        time_encoding_dim: int = 32,
        embedding_dim: int = 128,
        num_attention_heads: int = 4,
        num_gnn_layers: int = 2,
        edge_feature_dim: int = 16,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        dropout: float = 0.1,
        memory_updater: str = "gru",
        message_aggregator: str = "last",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.time_encoding_dim = time_encoding_dim
        self.edge_feature_dim = edge_feature_dim
        self.prediction_horizons = prediction_horizons
        self.message_aggregator = message_aggregator

        # 1. Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Feature gate — learned feature selection (TGIB-inspired)
        self.feature_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

        # 3. Time encoding
        self.time_encoder = TimeEncoding(time_encoding_dim)

        # 4. Memory module
        self.memory = MemoryModule(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_dim=embedding_dim,
            updater_type=memory_updater,
        )

        # 5. Time-aware memory-feature fusion
        # Input: memory (128) + features (128) + time encoding (32) = 288
        self.memory_fusion = nn.Sequential(
            nn.Linear(memory_dim + embedding_dim + time_encoding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

        # 6. Graph attention embedding layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                HeteroMessagePassingLayer(
                    in_dim=embedding_dim,
                    out_dim=embedding_dim,
                    edge_types=edge_types,
                    edge_dim=edge_feature_dim,
                    heads=num_attention_heads,
                    dropout=dropout,
                )
            )

        # 7. Temporal attention layer (was defined but never called before)
        self.temporal_attention = TemporalAttentionLayer(
            node_dim=embedding_dim,
            edge_dim=edge_feature_dim,
            time_dim=time_encoding_dim,
            output_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # 8. Multi-scale temporal context
        # Combines recent and older memory states for richer temporal signal
        self.temporal_context = nn.Sequential(
            nn.Linear(memory_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 9. Graph-level readout
        self.graph_readout = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention-weighted pooling
        self.pool_attention = nn.Linear(embedding_dim, 1)

        # 10. Multi-horizon prediction heads
        # Input: graph_emb (128) + max_emb (128) + temporal_ctx (128) = 384
        head_input = embedding_dim * 3
        self.prediction_heads = nn.ModuleDict()
        for h in prediction_horizons:
            self.prediction_heads[f"head_{h}h"] = nn.Sequential(
                nn.Linear(head_input, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(embedding_dim, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        # 11. Severity estimation head
        self.severity_head = nn.Sequential(
            nn.Linear(head_input, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

        # 12. Propagation path head (per-node cascade probability)
        self.propagation_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _build_neighbor_features(
        self,
        x: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor],
        edge_attr_dict: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather neighbor node features and edge features for temporal attention.

        Vectorized — no Python loops over edges.

        Returns:
            neighbor_features: [num_nodes, max_neighbors, embedding_dim]
            edge_features: [num_nodes, max_neighbors, edge_feature_dim]
        """
        device = x.device
        num_nodes = x.size(0)
        max_k = 8  # cap neighbors for efficiency

        # Collect all (dst, src) pairs and edge features across edge types
        all_dst = []
        all_src = []
        all_eattr = []

        for etype, ei in edge_index_dict.items():
            if ei.size(1) == 0:
                continue
            all_src.append(ei[0])
            all_dst.append(ei[1])
            ea = edge_attr_dict.get(etype) if edge_attr_dict else None
            if ea is not None and ea.size(0) == ei.size(1):
                all_eattr.append(ea)
            else:
                all_eattr.append(
                    torch.zeros(ei.size(1), self.edge_feature_dim, device=device)
                )

        if not all_dst:
            # No edges — return self-loops as dummy neighbors
            return (
                x.unsqueeze(1),  # [N, 1, emb]
                torch.zeros(num_nodes, 1, self.edge_feature_dim, device=device),
            )

        src_cat = torch.cat(all_src)       # [total_edges]
        dst_cat = torch.cat(all_dst)       # [total_edges]
        eattr_cat = torch.cat(all_eattr)   # [total_edges, edge_dim]

        # Count neighbors per destination node
        neighbor_count = torch.zeros(num_nodes, dtype=torch.long, device=device)
        neighbor_count.scatter_add_(0, dst_cat, torch.ones_like(dst_cat))
        actual_max_k = min(int(neighbor_count.max().item()), max_k)
        actual_max_k = max(actual_max_k, 1)

        neighbor_feats = torch.zeros(num_nodes, actual_max_k, self.embedding_dim, device=device)
        edge_feats = torch.zeros(num_nodes, actual_max_k, self.edge_feature_dim, device=device)

        # Vectorized slot filling: sort by destination, then assign positions
        sort_idx = torch.argsort(dst_cat)
        dst_sorted = dst_cat[sort_idx]
        src_sorted = src_cat[sort_idx]
        eattr_sorted = eattr_cat[sort_idx]

        # Compute position within each destination group
        # For each edge, its position = how many edges to the same dst came before it
        ones = torch.ones_like(dst_sorted)
        cumcount = torch.zeros_like(dst_sorted)
        for node_id in dst_sorted.unique():
            mask = dst_sorted == node_id
            cumcount[mask] = torch.arange(mask.sum(), device=device)

        # Only keep edges within the max_k budget
        valid = cumcount < actual_max_k
        if valid.any():
            d_valid = dst_sorted[valid]
            s_valid = src_sorted[valid]
            e_valid = eattr_sorted[valid]
            p_valid = cumcount[valid]
            neighbor_feats[d_valid, p_valid] = x[s_valid]
            edge_feats[d_valid, p_valid] = e_valid

        return neighbor_feats, edge_feats

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index_dict: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        edge_attr_dict: Optional[dict[str, torch.Tensor]] = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a single temporal graph snapshot.

        Args:
            node_features: [num_nodes, node_feature_dim].
            edge_index_dict: Dict of edge_type -> [2, num_edges].
            timestamps: [num_nodes] or scalar timestamp.
            edge_attr_dict: Optional edge features per type.
            return_embeddings: If True, also return node embeddings.

        Returns:
            Dict with keys:
              - cascade_{h}h: [1] cascade probability for each horizon h
              - severity: [1] estimated severity
              - propagation: [num_nodes, 1] per-node cascade probability
              - embeddings: (optional) [num_nodes, embedding_dim]
        """
        device = node_features.device

        # 1. Project input features
        x = self.input_proj(node_features)  # [num_nodes, embedding_dim]

        # 2. Apply feature gate (learned feature selection)
        gate = self.feature_gate(x)
        x = x * gate  # element-wise gating

        # 3. Compute time delta and time encoding
        if timestamps.dim() == 0:
            ts = timestamps.expand(self.num_nodes)
        else:
            ts = timestamps
        time_delta = ts - self.memory.last_update[:self.num_nodes].to(device)
        time_delta = time_delta.clamp(min=0)
        # Normalize to days (timestamps are Unix seconds — raw deltas ~1e9)
        time_delta = time_delta / 86400.0
        time_enc = self.time_encoder(time_delta)  # [num_nodes, time_encoding_dim]

        # 4. Time-aware memory fusion
        memory = self.memory.get_memory()  # [num_nodes, memory_dim]
        fused = self.memory_fusion(
            torch.cat([x, memory, time_enc], dim=-1)
        )  # [num_nodes, embedding_dim]

        # 5. Graph attention message passing
        x = fused
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index_dict, edge_attr_dict)

        # 6. Temporal attention over neighbors
        neighbor_feats, edge_feats = self._build_neighbor_features(
            x, edge_index_dict, edge_attr_dict
        )
        # Neighbor time encodings (use zeros — all from same snapshot)
        neighbor_time_enc = torch.zeros(
            self.num_nodes, neighbor_feats.size(1), self.time_encoding_dim,
            device=device
        )
        x = self.temporal_attention(
            query_node_features=x,
            neighbor_node_features=neighbor_feats,
            edge_features=edge_feats,
            time_encodings=time_enc,
            neighbor_time_encodings=neighbor_time_enc,
        )  # [num_nodes, embedding_dim]

        # 7. Update memory with new embeddings
        node_ids = torch.arange(self.num_nodes, device=device)
        self.memory.update_memory(node_ids, x, timestamps)

        # 8. Multi-scale temporal context from memory
        # Split memory into "recent" (updated this step) and "older" (previous state)
        current_memory = self.memory.get_memory()  # just-updated memory
        # Use the pre-update memory (approximated by the raw memory vector)
        # and current memory to capture multi-scale dynamics
        older_memory = memory.detach()  # pre-update memory snapshot
        temporal_ctx_input = torch.cat([current_memory, older_memory], dim=-1)
        temporal_ctx_per_node = self.temporal_context(
            temporal_ctx_input
        )  # [num_nodes, embedding_dim]

        # 9. Graph-level readout (attention-weighted pooling)
        node_embeddings = self.graph_readout(x)  # [num_nodes, embedding_dim]

        # Attention pooling
        attn_weights = F.softmax(
            self.pool_attention(node_embeddings), dim=0
        )  # [num_nodes, 1]
        graph_embedding = (attn_weights * node_embeddings).sum(
            dim=0, keepdim=True
        )  # [1, embedding_dim]

        # Max-pooled embedding
        max_embedding = node_embeddings.max(dim=0, keepdim=True).values

        # Pool temporal context to graph level
        temporal_ctx_graph = temporal_ctx_per_node.mean(
            dim=0, keepdim=True
        )  # [1, embedding_dim]

        # Concatenate: graph_emb + max_emb + temporal_ctx = 384
        combined = torch.cat(
            [graph_embedding, max_embedding, temporal_ctx_graph], dim=-1
        )  # [1, embedding_dim * 3]

        # 10. Multi-horizon predictions
        outputs = {}
        for h in self.prediction_horizons:
            logit = self.prediction_heads[f"head_{h}h"](combined)
            outputs[f"cascade_{h}h"] = logit.squeeze()

        # 11. Severity estimation
        outputs["severity"] = self.severity_head(combined).squeeze()

        # 12. Per-node propagation probability
        outputs["propagation"] = self.propagation_head(node_embeddings)

        if return_embeddings:
            outputs["embeddings"] = node_embeddings
            outputs["pool_attention"] = attn_weights.squeeze(-1)  # [num_nodes]

        return outputs

    def process_temporal_sequence(
        self,
        snapshots: list[dict],
        reset_memory: bool = True,
        tbptt_window: int = 10,
    ) -> list[dict[str, torch.Tensor]]:
        """Process a sequence of temporal graph snapshots.

        Args:
            snapshots: List of dicts with keys:
                node_features, edge_index_dict, timestamp, edge_attr_dict.
            reset_memory: Whether to reset memory at start.
            tbptt_window: Detach memory every N steps for TBPTT.

        Returns:
            List of prediction dicts, one per timestep.
        """
        if reset_memory:
            self.memory.reset_memory()

        all_outputs = []
        for i, snapshot in enumerate(snapshots):
            outputs = self.forward(
                node_features=snapshot["node_features"],
                edge_index_dict=snapshot["edge_index_dict"],
                timestamps=snapshot["timestamp"],
                edge_attr_dict=snapshot.get("edge_attr_dict"),
            )
            all_outputs.append(outputs)
            # TBPTT: only detach at window boundaries
            if (i + 1) % tbptt_window == 0:
                self.memory.detach_memory()

        return all_outputs

    def to(self, *args, **kwargs):
        """Override to() to also move memory tensors."""
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.memory._to_device(device)
        return result

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_memory(self):
        """Reset all memory to zeros."""
        self.memory.reset_memory()
