"""
LSTM baseline: processes temporal node features without graph structure.
Measures the contribution of the graph topology to prediction quality.
"""

import torch
import torch.nn as nn


class LSTMCascadePredictor(nn.Module):
    """LSTM baseline that processes concatenated protocol features as
    a multivariate time series, ignoring graph structure.

    Ablation: quantifies the value added by graph-based modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_nodes: int = 15,
        prediction_horizons: list[int] = [24, 72, 168, 720],
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.prediction_horizons = prediction_horizons

        # Flatten all node features into one vector per timestep
        total_input = input_dim * num_nodes

        self.input_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # LayerNorm on LSTM output stabilizes training
        self.output_norm = nn.LayerNorm(lstm_out_dim)

        # Prediction heads
        self.prediction_heads = nn.ModuleDict()
        for h in prediction_horizons:
            self.prediction_heads[f"head_{h}h"] = nn.Sequential(
                nn.Linear(lstm_out_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.severity_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feature_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            feature_sequence: [batch, seq_len, num_nodes, feature_dim]
                or [seq_len, num_nodes, feature_dim] (unbatched).

        Returns:
            Dict of predictions.
        """
        if feature_sequence.dim() == 3:
            feature_sequence = feature_sequence.unsqueeze(0)

        batch, seq_len, num_nodes, feat_dim = feature_sequence.shape

        # Flatten nodes into feature vector
        x = feature_sequence.view(batch, seq_len, -1)
        x = self.input_proj(x)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state with LayerNorm
        last_hidden = self.output_norm(lstm_out[:, -1, :])  # [batch, lstm_out_dim]

        outputs = {}
        for h in self.prediction_horizons:
            outputs[f"cascade_{h}h"] = self.prediction_heads[f"head_{h}h"](
                last_hidden
            ).squeeze()

        outputs["severity"] = self.severity_head(last_hidden).squeeze()
        return outputs
