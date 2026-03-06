"""
Memory module for the Temporal Graph Network.

Maintains per-node memory vectors that are updated at each interaction,
capturing long-term temporal patterns in the DeFi composability graph.

Key design: memory updates are NOT detached, so gradients flow through
the GRU across timesteps within a TBPTT window. detach_memory() is
called at window boundaries to truncate backpropagation.
"""

import torch
import torch.nn as nn
from typing import Optional


class MemoryModule(nn.Module):
    """GRU/RNN-based memory module that maintains per-node state vectors.

    Each protocol node has a persistent memory vector that gets updated
    when new events (TVL changes, liquidations, etc.) occur, allowing
    the model to capture temporal dynamics like building risk.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        message_dim: int,
        updater_type: str = "gru",
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph (protocols).
            memory_dim: Dimension of per-node memory vectors.
            message_dim: Dimension of incoming messages.
            updater_type: "gru" or "rnn".
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        # Memory storage — kept as a plain tensor (not buffer) so that
        # gradient-carrying tensors can be assigned without autograd issues.
        # Device management is handled explicitly in reset_memory / to().
        self.memory: torch.Tensor = torch.zeros(num_nodes, memory_dim)
        self.last_update: torch.Tensor = torch.zeros(num_nodes)

        # Memory updater
        if updater_type == "gru":
            self.updater = nn.GRUCell(message_dim, memory_dim)
        elif updater_type == "rnn":
            self.updater = nn.RNNCell(message_dim, memory_dim)
        else:
            raise ValueError(f"Unknown updater type: {updater_type}")

        # Message function: transforms raw events into messages
        self.message_fn = nn.Sequential(
            nn.Linear(memory_dim * 2 + message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def _to_device(self, device: torch.device):
        """Move memory tensors to specified device."""
        self.memory = self.memory.to(device)
        self.last_update = self.last_update.to(device)

    def get_memory(self, node_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retrieve current memory vectors.

        Returns memory WITH gradient history so that upstream computations
        can backpropagate through memory reads to previous updates.
        """
        if node_ids is None:
            return self.memory
        return self.memory[node_ids]

    def compute_messages(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages from source to target nodes."""
        source_memory = self.memory[source_ids]
        target_memory = self.memory[target_ids]

        msg_input = torch.cat(
            [source_memory, target_memory, edge_features], dim=-1
        )
        messages = self.message_fn(msg_input)
        return messages

    def aggregate_messages(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        aggregator: str = "last",
    ) -> torch.Tensor:
        """Aggregate messages for each node."""
        unique_nodes = node_ids.unique()
        aggregated = torch.zeros(
            len(unique_nodes), self.message_dim, device=messages.device
        )

        for i, node_id in enumerate(unique_nodes):
            mask = node_ids == node_id
            node_messages = messages[mask]
            if aggregator == "mean":
                aggregated[i] = node_messages.mean(dim=0)
            elif aggregator == "last":
                aggregated[i] = node_messages[-1]
            else:
                aggregated[i] = node_messages.mean(dim=0)

        return unique_nodes, aggregated

    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ):
        """Update memory vectors — gradients flow through for TBPTT.

        Creates a new tensor (no in-place modification) so that autograd
        can track the dependency chain across timesteps.
        """
        if len(node_ids) == 0:
            return

        current_memory = self.memory[node_ids]
        new_memory = self.updater(messages, current_memory)

        # Build updated memory WITHOUT in-place modification.
        # scatter new_memory into a fresh copy so autograd can track it.
        updated = self.memory.detach().clone()
        updated[node_ids] = new_memory
        self.memory = updated

        if timestamps is not None:
            self.last_update = self.last_update.detach().clone()
            self.last_update[node_ids] = timestamps.detach()

    def reset_memory(self, node_ids: Optional[torch.Tensor] = None):
        """Reset memory to zeros (detached)."""
        device = self.memory.device
        if node_ids is None:
            self.memory = torch.zeros(
                self.num_nodes, self.memory_dim, device=device
            )
            self.last_update = torch.zeros(self.num_nodes, device=device)
        else:
            self.memory = self.memory.detach().clone()
            self.memory[node_ids] = 0.0
            self.last_update = self.last_update.detach().clone()
            self.last_update[node_ids] = 0.0

    def detach_memory(self):
        """Detach memory from computation graph (TBPTT boundary)."""
        self.memory = self.memory.detach()
        self.last_update = self.last_update.detach()
