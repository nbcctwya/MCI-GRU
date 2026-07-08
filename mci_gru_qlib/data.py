"""Data pipeline: qlib OHLCV -> standardized feature tensors + forward-rank labels.

Reuses the per-day processing logic of the original ``code/csi300.py``
(3-sigma clipping + z-score, cross-sectionally within each trading day) and the
forward-return rank label of ``fun_label``.

Output tensors are organized day-major: for each trading day we materialize a
full cross-section of all universe stocks so the graph couples them together in
one forward pass (batch_size == 1 day, matching the original design).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# qlib must be initialized (see ``init_qlib``) before any ``D`` call.
import qlib
from qlib.data import D


# --------------------------------------------------------------------------- #
# Reused processing primitives (verbatim logic from code/csi300.py:44-63).
# --------------------------------------------------------------------------- #
def filter_extreme_3sigma(series: pd.Series, n: int = 3) -> pd.Series:
    mean = series.mean()
    std = series.std()
    return np.clip(series, mean - n * std, mean + n * std)


def standardize_zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    mean = series.mean()
    return (series - mean) / std


def process_day_features(day_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Per-day: fill NaN with cross-sectional mean, then 0, then 3-sigma + z-score.

    Mirrors code/csi300.py:262-272 + ``process_daily_df_std``.
    """
    day_df = day_df.copy()
    for c in feature_cols:
        mean_val = day_df[c].mean()
        day_df[c] = day_df[c].fillna(mean_val)
    day_df = day_df.fillna(0.0)
    for c in feature_cols:
        day_df[c] = filter_extreme_3sigma(day_df[c])
        day_df[c] = standardize_zscore(day_df[c])
    return day_df


@dataclass
class SplitData:
    dates: List[pd.Timestamp]            # [D] trading days
    x_ts: np.ndarray                     # [D, N, his_t, F]
    x_graph: np.ndarray                  # [D, N, F]
    y: np.ndarray                        # [D, N] cross-sectional rank (training target)
    raw_y: np.ndarray                    # [D, N] raw forward return (for IC)


@dataclass
class PreparedData:
    universe: List[str]
    features: List[str]
    splits: Dict[str, SplitData] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def init_qlib(cfg) -> None:
    qlib.init(provider_uri=cfg.provider_uri, region=cfg.region)


def load_raw_panel(cfg) -> pd.DataFrame:
    """Load OHLCV via qlib; add proxy turnover if requested.

    Returns a long DataFrame indexed (instrument, datetime) with columns
    ``close, open, high, low, volume`` (and ``turnover`` if proxied). ``close``
    is the qlib adjusted close.
    """
    fields = ["$open", "$close", "$high", "$low", "$volume"]
    df = D.features(
        D.instruments(market=cfg.market),
        fields,
        start_time=cfg.universe.start,
        end_time=cfg.universe.end,
        freq="day",
    )
    df = df.rename(columns={c: c.lstrip("$") for c in df.columns})
    df.index.set_names(["instrument", "datetime"], inplace=True)

    if cfg.turnover_mode == "proxy":
        df["turnover"] = df["volume"] * df["close"]
    return df


def select_universe(panel: pd.DataFrame, cfg) -> List[str]:
    """Instruments present on >= presence_threshold of trading days (deterministic sort).

    For smoke runs (``max_instruments`` set) we skip the presence filter and
    take the first N instruments that have data, so a tiny universe still forms
    a graph.
    """
    if cfg.universe.max_instruments:
        present = panel["close"].dropna().groupby(level="instrument").size()
        present = present[present > 0].sort_index()
        return list(present.index[: cfg.universe.max_instruments])

    # qlib calendar over the universe window defines the denominator.
    calendar = D.calendar(start_time=cfg.universe.start, end_time=cfg.universe.end)
    n_days = len(calendar)
    counts = panel["close"].groupby(level="instrument").size()  # non-NaN rows per inst
    keep = counts[counts >= cfg.universe.presence_threshold * n_days]
    return sorted(keep.index)


def prepare_data(cfg) -> PreparedData:
    """End-to-end: load -> universe -> standardize -> windows -> labels -> splits."""
    features: List[str] = list(cfg.features)
    raw = load_raw_panel(cfg)
    universe = select_universe(raw, cfg)

    # Restrict to universe and switch to (datetime, instrument) ordering.
    raw_u = raw.loc[universe].sort_index()
    raw_u = raw_u.swaplevel().sort_index()  # -> (datetime, instrument)
    raw_u = raw_u[~raw_u.index.duplicated(keep="last")]

    # Per-day cross-sectional standardization of features (code/csi300.py logic).
    processed_parts = []
    for dt, day_df in raw_u.groupby(level="datetime"):
        processed_parts.append(process_day_features(day_df, features))
    proc = pd.concat(processed_parts)
    proc = proc.sort_index()

    # Forward-return rank label from adjusted close (fun_label, label_t horizon).
    close = raw_u["close"].unstack(level="instrument")  # (datetime, instrument)
    t1 = close.shift(-1)
    tn = close.shift(-cfg.model.label_t)
    fwd_ret = (tn / t1 - 1.0)
    # Cross-sectional pct rank per day (ascending=True -> higher score = better).
    label = fwd_ret.rank(axis=1, ascending=True, pct=True)

    # Align everything onto a common (datetime, instrument) grid.
    all_dates = close.index
    n = len(universe)
    f = len(features)

    feat_mat = np.zeros((len(all_dates), n, f), dtype=np.float32)
    for k, feat in enumerate(features):
        mat = proc[feat].unstack(level="instrument").reindex(index=all_dates, columns=universe)
        feat_mat[:, :, k] = mat.values
    feat_mat = np.nan_to_num(feat_mat, nan=0.0)

    label_mat = label.reindex(index=all_dates, columns=universe).values.astype(np.float32)
    raw_ret_mat = fwd_ret.reindex(index=all_dates, columns=universe).values.astype(np.float32)

    his_t = cfg.model.his_t
    prepared = PreparedData(universe=universe, features=features)
    for name in ("train", "valid", "test"):
        split = cfg.splits[name]
        d0 = pd.Timestamp(split.start)
        d1 = pd.Timestamp(split.end)
        pos = np.where((all_dates >= d0) & (all_dates <= d1))[0]

        days_idx, x_ts_list, x_graph_list, y_list, raw_y_list = [], [], [], [], []
        for i in pos:
            if i < his_t - 1:                     # not enough history for a window
                continue
            y_vec = label_mat[i]
            if not np.isfinite(y_vec).any():      # no realizable label this day
                continue
            window = feat_mat[i - his_t + 1 : i + 1]      # (his_t, N, F)
            x_ts_list.append(np.transpose(window, (1, 0, 2)))   # (N, his_t, F)
            x_graph_list.append(feat_mat[i])              # (N, F)
            y_list.append(np.nan_to_num(y_vec, nan=0.0))
            raw_y_list.append(np.nan_to_num(raw_ret_mat[i], nan=0.0))
            days_idx.append(all_dates[i])

        if not x_ts_list:
            raise RuntimeError(
                f"No usable days in split '{name}' [{split.start}..{split.end}]; "
                f"check date range / universe size."
            )
        prepared.splits[name] = SplitData(
            dates=days_idx,
            x_ts=np.stack(x_ts_list, axis=0),
            x_graph=np.stack(x_graph_list, axis=0),
            y=np.stack(y_list, axis=0),
            raw_y=np.stack(raw_y_list, axis=0),
        )
    return prepared
