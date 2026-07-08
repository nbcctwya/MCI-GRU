"""Train-only stock-correlation graph (replaces torch_geometric dependency).

Reuses the logic of ``fun_relation`` + ``fun_graph`` from ``code/csi300.py``:
the Pearson correlation of daily returns over the last ``corr_window_days`` of
the *training* period defines the graph; pairs with correlation above
``judge_value`` become undirected (bidirectional) edges whose weight is the
correlation. The graph is built from train data only (no valid/test leakage),
cached per market.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qlib.data import D


def build_graph(cfg, universe):
    """Return (edge_index[2,E] long, edge_weight[E,1] float) over the universe."""
    train_end = pd.Timestamp(cfg.splits.train.end)
    calendar = D.calendar(end_time=str(train_end.date()))
    window = calendar[-cfg.graph.corr_window_days :]
    assert calendar[-1] <= train_end + pd.Timedelta(days=1), "graph must use train data only"

    df = D.features(universe, ["$close"], start_time=window[0], end_time=window[-1], freq="day")
    close = df["$close"].unstack(level="instrument").reindex(columns=universe)
    rets = close / close.shift(1) - 1.0
    corr = rets.corr().reindex(index=universe, columns=universe)

    mat = corr.values
    n = len(universe)
    judge = cfg.graph.judge_value

    src, dst, w = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            wij = mat[i, j]
            if np.isfinite(wij) and wij > judge:
                src += [i, j]
                dst += [j, i]
                w += [wij, wij]
    if not src:  # pragma: no cover - degenerate; smoke uses judge_value=0.3 to avoid
        src = list(range(n))
        dst = list(range(n))
        w = [1.0] * n

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float).unsqueeze(1)  # (E,1) for edge_dim=1
    return edge_index, edge_weight


def build_or_load_graph(cfg, universe):
    cache = Path(cfg.output_dir) / "graph.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = torch.load(cache, weights_only=False)
        return data["edge_index"], data["edge_weight"]
    edge_index, edge_weight = build_graph(cfg, universe)
    torch.save({"edge_index": edge_index, "edge_weight": edge_weight}, cache)
    return edge_index, edge_weight
