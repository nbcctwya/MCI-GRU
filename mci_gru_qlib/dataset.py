"""Day-keyed dataset (fixes Bug A in the original).

The original ``model_data`` used two independently-shuffled ``DataLoader``s
(time-series and graph) zipped positionally, so a day's time-series window could
be paired with a *different* day's graph snapshot and label. Here one dataset
item is exactly one trading day and yields the window, the day's graph snapshot
and the label **from the same day**, with the shared graph structure exposed as
an attribute.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .data import SplitData


class CombinedDayDataset(Dataset):
    def __init__(self, split: SplitData, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self.x_ts = torch.from_numpy(split.x_ts)        # (D, N, his_t, F)
        self.x_graph = torch.from_numpy(split.x_graph)  # (D, N, F)
        self.y = torch.from_numpy(split.y)              # (D, N) rank target
        self.raw_y = torch.from_numpy(split.raw_y)      # (D, N) raw forward return (for IC)
        self.dates = list(split.dates)                  # [D] pd.Timestamp
        self.edge_index = edge_index                    # (2, E) shared
        self.edge_weight = edge_weight                  # (E, 1) shared

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, i):
        return self.x_ts[i], self.x_graph[i], self.y[i]
