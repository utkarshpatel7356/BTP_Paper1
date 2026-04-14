"""
src/models/gat_model.py
-----------------------
Graph Attention Network (GAT) + Temporal GRU encoder.

Architecture:
  Input : (N+1, T, F)  node feature sequences
  1. For each time step t in [t-W, t]:
       h_t = GAT(x_t, edge_index)    -> (N+1, hidden)
  2. Stack h over the window W:
       H = [h_{t-W}, ..., h_t]       -> (N+1, W, hidden)
  3. GRU over time dimension:
       z = GRU(H)[-1]                -> (N+1, hidden)
  4. Index prediction head:
       y_hat = MLP(z[index_node])     -> scalar

The node embeddings z are used downstream for influence scoring.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATTemporalEncoder(nn.Module):
    """
    Per-timestep GAT followed by a GRU over the temporal window.

    Parameters
    ----------
    in_features  : F — input feature dimension per node per timestep
    hidden       : hidden dimension (also output embedding dim)
    heads        : number of GAT attention heads
    num_layers   : number of stacked GAT layers
    gru_window   : number of timesteps fed to GRU
    dropout      : dropout probability
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        heads: int = 4,
        num_layers: int = 2,
        gru_window: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru_window = gru_window
        self.hidden = hidden

        # Stack of GAT layers
        self.gat_layers = nn.ModuleList()
        for layer in range(num_layers):
            in_ch = in_features if layer == 0 else hidden
            concat = layer < num_layers - 1  # concat heads on all but last
            out_ch = hidden // heads if concat else hidden
            self.gat_layers.append(
                GATConv(
                    in_ch,
                    out_ch,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

        self.gat_norm = nn.LayerNorm(hidden)

        # GRU over temporal dimension
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def _gat_forward(
        self,
        x_t: torch.Tensor,        # (N+1, F)
        edge_index: torch.Tensor,  # (2, E)
        return_attn: bool = False,
    ):
        """Apply all GAT layers to a single timestep."""
        h = x_t
        attn_weights_last = None
        edge_index_last = None

        for i, layer in enumerate(self.gat_layers):
            if i < len(self.gat_layers) - 1:
                h = layer(h, edge_index)
            else:
                # Return attention weights from the last layer
                h, (ei_out, attn) = layer(h, edge_index, return_attention_weights=True)
                if return_attn:
                    attn_weights_last = attn   # (E_with_self_loops, heads)
                    edge_index_last = ei_out   # (2, E_with_self_loops)
            h = self.act(h)
            h = self.dropout(h)
        h = self.gat_norm(h)
        return h, attn_weights_last, edge_index_last

    def forward(
        self,
        x: torch.Tensor,          # (N+1, T, F)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor = None,  # unused — GAT learns its own attention
    ) -> torch.Tensor:
        """
        Returns
        -------
        embeddings       : (N+1, hidden) — one embedding per node after GRU
        attn_weights     : (E', heads) — saved attention weights from last GAT layer
                           E' includes self-loops added by GATConv
        attn_edge_index  : (2, E') — edge index corresponding to attention weights
        """
        N_plus_1, T, F = x.shape
        W = min(self.gru_window, T)

        # Apply GAT at each timestep to get h_t for all nodes
        gat_outs = []  # will be (W, N+1, hidden)
        attn_weights_last = None
        attn_edge_index_last = None

        for t in range(max(0, T - W), T):
            x_t = x[:, t, :]  # (N+1, F)
            is_last = (t == T - 1)
            h, attn, ei = self._gat_forward(x_t, edge_index, return_attn=is_last)
            if is_last:
                attn_weights_last = attn
                attn_edge_index_last = ei
            gat_outs.append(h)  # (N+1, hidden)

        # Stack to (N+1, W, hidden) for GRU
        H = torch.stack(gat_outs, dim=1)  # (N+1, W, hidden)
        out, _ = self.gru(H)              # (N+1, W, hidden)
        embeddings = out[:, -1, :]        # (N+1, hidden) — last timestep

        return embeddings, attn_weights_last, attn_edge_index_last


class SparseIndexGNN(nn.Module):
    """
    Full model: encoder + index return prediction head.

    Parameters match GATTemporalEncoder. The prediction head maps
    the index sink node's embedding to a scalar (daily return).
    """

    def __init__(self, in_features, hidden=64, heads=4, num_layers=2, gru_window=60, dropout=0.1):
        super().__init__()
        self.encoder = GATTemporalEncoder(in_features, hidden, heads, num_layers, gru_window, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        x            : (N+1, T, F)
        edge_index   : (2, E)
        edge_weight  : (E,) — unused, kept for API compatibility

        Returns
        -------
        y_hat             : scalar — predicted index return at time T
        embeddings        : (N+1, hidden) — use for influence scoring
        attn_weights      : (E', heads) — attention from last GAT layer
        attn_edge_index   : (2, E') — corresponding edge index (includes self-loops)
        """
        embeddings, attn_weights, attn_edge_index = self.encoder(x, edge_index, edge_weight)
        y_hat = self.head(embeddings[-1]).squeeze()  # last node = index sink
        return y_hat, embeddings, attn_weights, attn_edge_index
