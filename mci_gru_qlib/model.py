"""MCI-GRU model, ported from ``code/csi300.py``.

Two changes vs. the original:
  * **Bug B fix**: ``StockPredictionModel.forward`` used ``h_gru[-1, :, :]``
    which indexes the *batch* dimension (only valid at batch_size==1). We now
    assert batch_size==1 and squeeze it explicitly.
  * ``torch_geometric.nn.GATConv`` is replaced by a pure-PyTorch ``GATConvTorch``
    so the baseline has no ``torch_geometric`` dependency. The layer structure
    (2 conv layers, multi-head, edge weight as an edge feature, self-loops) and
    all in/out dimensions are preserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Pure-PyTorch GAT convolution (drop-in for torch_geometric.nn.GATConv).
# --------------------------------------------------------------------------- #
def _scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # src: (E, ...); index: (E,). Returns (dim_size, ...).
    flat = src.reshape(src.size(0), -1)
    out = torch.full((dim_size, flat.size(1)), float("-inf"), device=src.device, dtype=src.dtype)
    out.scatter_reduce_(0, index.unsqueeze(1).expand_as(flat), flat, reduce="amax", include_self=True)
    return out.reshape(dim_size, *src.shape[1:])


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # src: (E, ...); index: (E,). Returns (dim_size, ...).
    flat = src.reshape(src.size(0), -1)
    out = torch.zeros(dim_size, flat.size(1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, flat)
    return out.reshape(dim_size, *src.shape[1:])


class GATConvTorch(nn.Module):
    """Multi-head GAT convolution mirroring PyG ``GATConv`` semantics.

    - ``heads`` attention heads; if ``concat`` the output is ``heads*out_channels``,
      else the mean over heads.
    - ``edge_dim`` edge-feature width; an edge feature (here the correlation
      weight) contributes additively to the attention logits.
    - self-loops are added by default.
    """

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 edge_dim=1, negative_slope=0.2, add_self_loops=True, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.edge_dim is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        N = x.size(0)
        H, C = self.heads, self.out_channels
        x_proj = self.lin(x).view(N, H, C)  # (N,H,C)

        if self.add_self_loops:
            self_idx = torch.arange(N, device=x.device).view(1, N).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_idx], dim=1)
            if self.edge_dim is not None:
                sl_attr = torch.ones(N, self.edge_dim, device=x.device, dtype=x.dtype)
                edge_attr = torch.cat([edge_attr, sl_attr], dim=0)

        src, dst = edge_index[0], edge_index[1]
        # Attention logits (E,H).
        alpha = (x_proj[src] * self.att_src).sum(-1) + (x_proj[dst] * self.att_dst).sum(-1)
        if self.edge_dim is not None:
            edge_proj = self.lin_edge(edge_attr).view(-1, H, C)  # (E,H,C)
            alpha = alpha + (edge_proj * self.att_edge).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Softmax over each destination node's incoming edges, per head.
        alpha = alpha - _scatter_max(alpha, dst, N)[dst]
        alpha_exp = alpha.exp()
        alpha_sum = _scatter_add(alpha_exp, dst, N)[dst].clamp_min(1e-16)
        alpha = alpha_exp / alpha_sum  # (E,H)

        # Weighted aggregation of source node features.
        msg = x_proj[src] * alpha.unsqueeze(-1)  # (E,H,C)
        out = _scatter_add(msg, dst, N)          # (N,H,C)

        if self.concat:
            out = out.reshape(N, H * C)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------------------------------------------------------------- #
# GAT layer wrappers (identical dims to code/csi300.py:337-359).
# --------------------------------------------------------------------------- #
class GATLayer(nn.Module):
    def __init__(self, hidden_size_gat1, output_gat1, in_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATConvTorch(in_channels, hidden_size_gat1, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConvTorch(hidden_size_gat1 * heads, output_gat1, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


class GATLayer_1(nn.Module):
    def __init__(self, hidden_size_gat2, in_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATConvTorch(in_channels, hidden_size_gat2, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConvTorch(hidden_size_gat2 * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


# --------------------------------------------------------------------------- #
# Attention blocks (ported verbatim from code/csi300.py:312-402).
# --------------------------------------------------------------------------- #
class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(input_size, hidden_size * 2, bias=False)
        self.w_hh = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.attention = nn.Linear(hidden_size, input_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        attn_scores = self.attention(hidden)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = x * attn_weights
        gates = self.w_ih(x) + self.w_hh(hidden)
        r_gate, u_gate = gates.chunk(2, 2)
        r_gate = torch.sigmoid(r_gate)
        u_gate = torch.sigmoid(u_gate)
        h_hat = self.tanh(r_gate * hidden)
        new_hidden = u_gate * hidden + (1 - u_gate) * h_hat
        return new_hidden


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_gat1, output_gat1,
                 gat_in_channels, gat_out_channels, gat_heads, hidden_size_gat2,
                 embed_dim, num_hidden_states):
        super().__init__()
        self.attention_gru = AttentionGRUCell(input_size, hidden_size)
        self.gat_layer = GATLayer(hidden_size_gat1, output_gat1, gat_in_channels, gat_out_channels, gat_heads)
        self.cross_attention = CrossAttention(hidden_size)
        self.num_hidden_states = num_hidden_states
        self.market_hidden_states_1 = nn.Parameter(torch.randn(num_hidden_states, hidden_size))
        self.market_hidden_states_2 = nn.Parameter(torch.randn(num_hidden_states, hidden_size))
        self.self_attention = SelfAttention(hidden_size * 4)
        self.final_gat = GATLayer_1(hidden_size_gat2, hidden_size * 4, 1)
        # NOTE: the original applies a final ReLU here. We drop it: for the larger
        # SP500 config (hidden=256) the final GAT output is systematically negative
        # at init, so ReLU zeroes everything -> dead network (zero gradient, flat
        # loss). ReLU is unnecessary for a cross-sectional ranker scored by IC /
        # top-K backtest (only relative order matters) and MSE accepts any real
        # output. See README_baseline.md "Design notes".

    def forward(self, x_time_series, x_graph, edge_index, edge_weight):
        batch_size, num_samples, num_time_steps, num_features = x_time_series.size()
        assert batch_size == 1, "MCI-GRU processes one trading day per forward pass (batch_size=1)."

        h_gru = torch.zeros(batch_size, num_samples, self.attention_gru.hidden_size, device=x_time_series.device)
        for t in range(num_time_steps):
            h_gru = self.attention_gru(x_time_series[:, :, t, :], h_gru)
        h_gru_1 = h_gru.squeeze(0)                       # (N, hidden)  [Bug B fix]

        x_gat = self.gat_layer(x_graph, edge_index, edge_weight)  # (N, hidden)

        stock_rep_1 = self.cross_attention(
            h_gru_1.unsqueeze(1), self.market_hidden_states_1, self.market_hidden_states_1
        ).squeeze(1)
        stock_rep_2 = self.cross_attention(
            x_gat.unsqueeze(1), self.market_hidden_states_2, self.market_hidden_states_2
        ).squeeze(1)

        concatenated = torch.cat([h_gru_1, x_gat, stock_rep_1, stock_rep_2], dim=1)  # (N, 4*hidden)
        attention_output = self.self_attention(concatenated.unsqueeze(1)).squeeze(1)
        out = self.final_gat(attention_output, edge_index, edge_weight)  # (N, 1)
        return out.squeeze(1)  # (N,)  -- no final activation (see __init__ note)


def build_model(cfg) -> StockPredictionModel:
    m = cfg.model
    return StockPredictionModel(
        input_size=len(cfg.features),
        hidden_size=m.hidden,
        hidden_size_gat1=m.hidden_size_gat1,
        output_gat1=m.output_gat1,
        gat_in_channels=len(cfg.features),
        gat_out_channels=m.gat_out_channels,
        gat_heads=m.gat_heads,
        hidden_size_gat2=m.hidden_size_gat2,
        embed_dim=m.embed_dim,
        num_hidden_states=m.num_hidden_states,
    )
