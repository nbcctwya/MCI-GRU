"""Training: per-seed train/valid loop with best-valid (RankIC) checkpoint
selection, then test-set prediction.

Mirrors the original setup (Adam, MSE, day-level batches) but adds:
  * an explicit validation segment for model selection (the original had none),
  * deterministic multi-seed control,
  * test predictions returned as a qlib-compatible ``(datetime, instrument)``
    DataFrame with a ``score`` column.
"""
from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from qlib.contrib.eva.alpha import calc_ic

from .model import build_model
from .dataset import CombinedDayDataset


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg) -> torch.device:
    dev = cfg.train.device
    if dev == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _predict(model, dataset: CombinedDayDataset, device: torch.device):
    """Return (scores[D,N], raw_y[D,N], dates[D])."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    ei = dataset.edge_index.to(device)
    ew = dataset.edge_weight.to(device)
    scores, raws, dates = [], [], []
    with torch.no_grad():
        for i, (x_ts, x_graph, _y) in enumerate(loader):
            x_ts = x_ts.to(device)
            x_graph = x_graph.squeeze(0).to(device)  # (N,F); loader adds a batch dim
            out = model(x_ts, x_graph, ei, ew).cpu().numpy()
            scores.append(out)
            raws.append(dataset.raw_y[i])
            dates.append(dataset.dates[i])
    return scores, raws, dates


def _scores_to_df(scores, raws, dates, universe) -> pd.DataFrame:
    n = len(universe)
    inst = np.asarray(universe)
    dt = np.asarray(dates)
    idx = pd.MultiIndex.from_arrays(
        [np.repeat(dt, n), np.tile(inst, len(dates))], names=["datetime", "instrument"]
    )
    return pd.DataFrame(
        {"score": np.concatenate(scores).reshape(-1),
         "label": np.concatenate(raws).reshape(-1)},
        index=idx,
    )


def _valid_metrics(model, valid_ds, universe, device) -> Tuple[float, float]:
    scores, raws, dates = _predict(model, valid_ds, device)
    df = _scores_to_df(scores, raws, dates, universe)
    _ic, ric = calc_ic(df["score"], df["label"])
    rankic = float(np.nan_to_num(ric.mean()))
    # MSE against the rank target stored on the dataset.
    pred = torch.from_numpy(np.concatenate(scores))
    target = valid_ds.y.reshape(-1)
    mse = float(torch.nn.functional.mse_loss(pred, target).item())
    return rankic, mse


def train_one_seed(cfg, prepared, graph, seed, device, ckpt_path):
    set_seed(seed)
    train_ds = CombinedDayDataset(prepared.splits["train"], *graph)
    valid_ds = CombinedDayDataset(prepared.splits["valid"], *graph)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    model = build_model(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)

    ei = train_ds.edge_index.to(device)
    ew = train_ds.edge_weight.to(device)
    select = cfg.train.select_metric

    best_score = -float("inf")
    best_state = None
    for epoch in range(cfg.train.num_epochs):
        model.train()
        running = 0.0
        for x_ts, x_graph, y in train_loader:
            x_ts = x_ts.to(device)
            x_graph = x_graph.squeeze(0).to(device)   # (N,F); loader adds a batch dim
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x_ts, x_graph, ei, ew)        # (N,)
            loss = criterion(out, y.view(-1))
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        rankic, vmse = _valid_metrics(model, valid_ds, prepared.universe, device)
        score = rankic if select == "valid_rankic" else -vmse
        improved = score > best_score
        tag = " *" if improved else ""
        print(f"  [seed {seed}] epoch {epoch + 1}/{cfg.train.num_epochs} "
              f"train_loss={train_loss:.4f} valid_rankic={rankic:.4f} valid_mse={vmse:.4f}{tag}",
              flush=True)
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # Predict on the test segment with the best-valid checkpoint.
    test_ds = CombinedDayDataset(prepared.splits["test"], *graph)
    scores, raws, dates = _predict(model, test_ds, device)
    pred_df = _scores_to_df(scores, raws, dates, prepared.universe)  # cols: score, label
    _ic, ric = calc_ic(pred_df["score"], pred_df["label"])
    test_rankic = float(np.nan_to_num(ric.mean()))
    return pred_df, {"valid_best": best_score, "test_rankic": test_rankic}
