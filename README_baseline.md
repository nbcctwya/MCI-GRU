# MCI-GRU qlib Baseline

Reproduction of **MCI-GRU** (Neurocomputing 2025, [arXiv:2410.20679](https://arxiv.org/abs/2410.20679)) as a **paper baseline**, fed by a local **qlib** data store and backtested with **qlib**. The original authors' scripts live in `code/` (rolling monthly windows, private CSV, `torch_geometric`); this package restructures the model into a clean train/valid/test pipeline with qlib backtest and multi-seed aggregation.

## Setup

```bash
conda activate mci-gru          # python 3.9, torch 2.8, qlib 0.9.7 already present
# qlib data is expected at:
#   CN: ~/.qlib/qlib_data/cn_data     (region cn)
#   US: ~/.qlib/qlib_data/us_data     (region us)
pip install -r requirements.txt  # pyqlib + pyyaml (+ existing torch/numpy/pandas)
```

No `torch_geometric` is required — the GAT layers are reimplemented in pure PyTorch (`mci_gru_qlib/model.py`).

## Markets, split, hyperparameters

| | CSI 300 (CN) | S&P 500 (US) |
|---|---|---|
| qlib region / pool | `cn` / `csi300` | `us` / `sp500` |
| features | close, open, high, low, **turnover (proxy = volume×close)**, volume | close, open, high, low, volume (no turnover) |
| model hidden / market-states / epochs | 32 / 16 / 20 | 256 / 4 / 5 |
| benchmark | `SH000300` | `^gspc` |

- **Split**: train 2009-01-01..2020-12-31 · valid 2021-01-01..2022-12-31 · test 2023-01-01..2025-12-31.
- **Universe**: stocks present on ≥95% of trading days in 2009–2025 (per market).
- **Training**: Adam (lr 1e-3), MSE, his_t=10 history window, label_t=5 forward return → cross-sectional rank. Best-valid (RankIC) checkpoint, 1 model per seed.
- **Seeds**: 0–4; report mean ± std.
- **Backtest**: `TopkDropoutStrategy`, topk=30, n_drop=5, daily, account 1e8; costs **buy `open_cost=0.0005` / sell `close_cost=0.0015`** (fractions); A-share `limit_threshold=0.095`, US `None`.

Each market uses the **original repo's per-market defaults** (`code/csi300.py`, `code/sp500.py`).

## Run

```bash
# Smoke test (tiny, CPU, <2 min each) — verifies the full pipeline end-to-end:
python -m mci_gru_qlib.run --config configs/csi300.yaml --smoke
python -m mci_gru_qlib.run --config configs/sp500.yaml --smoke

# Real run (5 seeds, GPU recommended) — run when the machine is free:
python -m mci_gru_qlib.run --config configs/csi300.yaml
python -m mci_gru_qlib.run --config configs/sp500.yaml
# fewer seeds for a quick check:  --seeds 0
```

## Outputs

```
outputs/<market>/
  graph.pt                         # cached correlation graph (train-only)
  seed<0..4>/
    best.pth                       # best-valid checkpoint
    pred_test.parquet              # (datetime, instrument) -> score[, label]
    report.parquet                 # qlib daily backtest report
    metrics.json                   # per-seed metrics
  per_seed.csv                     # per-seed metric table
  summary.csv / summary.json       # mean +/- std over seeds
outputs/<market>_smoke/ ...        # --smoke outputs
```

Reported metrics: `annualized_return`, `excess_annualized_return` (vs benchmark), `excess_net_annualized_return` (after cost), `information_ratio`, `max_drawdown`, `IC`, `RankIC`, `ICIR`, `RankICIR`.

## Design notes (vs. original)

- **qlib data adapter** (`data.py`): loads OHLCV via `D.features`; the original `$turnover` is 100% NaN in qlib, so CSI300 proxies `turnover = volume × close` (z-scored → scale-invariant); SP500 matches the original 5-feature setup. Prices are qlib-adjusted (correct for return labels & backtest).
- **Fixed universe**: the original required stocks present every day in a ~3-month window; over 17 years that survives almost nothing. We instead keep ≥95%-presence stocks → stable universe for the fixed graph.
- **Train-only graph** (`graph.py`): correlation matrix over the last 250 train days only (no valid/test leakage), cached per market.
- **Two bug fixes**:
  - *Data pairing* (`code/csi300.py` `model_data`): the original zipped two independently-shuffled loaders, pairing a day's time-series with a different day's graph/label. Fixed by one day-keyed `CombinedDayDataset`.
  - *Batch indexing* (`StockPredictionModel.forward`): `h_gru[-1,:,:]` indexed the batch dim. Fixed to `squeeze(0)` with a batch_size=1 assertion.
- **Pure-PyTorch GAT** (`GATConvTorch`): drop-in for `torch_geometric.nn.GATConv` (multi-head, edge-weight feature, self-loops), preserving the original layer dims — no `torch_geometric` install.
- Numbers will **not** match the paper bit-for-bit (different data source, longer horizon, proxy turnover, fixed universe/split); the architecture, losses, and features are faithful.

## Package layout

```
mci_gru_qlib/
  config.py    load YAML -> Namespace; --smoke overrides
  data.py      qlib OHLCV, proxy turnover, >=95% universe, per-day 3sigma+zscore, windows, forward-rank labels
  graph.py     train-only correlation graph (cached)
  model.py     AttentionGRUCell, GATConvTorch, GATLayer(_1), Cross/SelfAttention, StockPredictionModel
  dataset.py   CombinedDayDataset (one day = window+graph+label, same day)
  train.py     seed control, train/valid loop, best-valid(RankIC) selection, test prediction
  backtest.py  qlib backtest_daily + risk_analysis + calc_ic, region-aware costs/benchmark
  run.py       CLI orchestrator (market x seed -> mean+/-std summary)
configs/csi300.yaml, configs/sp500.yaml
```
