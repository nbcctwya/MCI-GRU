"""CLI orchestrator for the MCI-GRU qlib baseline.

Usage:
    python -m mci_gru_qlib.run --config configs/csi300.yaml [--smoke] [--seeds 0,1,2,3,4]

Flow: load config -> init qlib -> prepare data -> build graph (cached) ->
loop seeds {train; predict; backtest} -> aggregate mean+std -> summary.csv/json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import init_qlib, prepare_data
from .graph import build_or_load_graph
from .train import train_one_seed, resolve_device
from .backtest import run_backtest


SUMMARY_METRICS = [
    "annualized_return",
    "excess_annualized_return",
    "excess_net_annualized_return",
    "information_ratio",
    "max_drawdown",
    "IC",
    "RankIC",
    "ICIR",
    "RankICIR",
    "test_rankic",
]


def summarize(rows):
    df = pd.DataFrame(rows)
    out = {}
    for k in SUMMARY_METRICS:
        if k in df.columns:
            out[k] = {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=0))}
    return out


def _smoke_asserts(cfg, prepared, pred_df, metrics):
    test = prepared.splits["test"]
    n = len(prepared.universe)
    assert pred_df.index.names == ["datetime", "instrument"], pred_df.index.names
    assert "score" in pred_df.columns
    assert len(pred_df) == len(test.dates) * n, (len(pred_df), len(test.dates), n)
    for col in ["annualized_return", "information_ratio", "max_drawdown", "RankIC"]:
        assert col in metrics, f"missing metric {col}"
    print("\n[smoke] OK: pred=%s universe=%d metrics=%d" % (pred_df.shape, n, len(metrics)))


def main():
    ap = argparse.ArgumentParser(description="MCI-GRU qlib baseline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (CPU)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (overrides config)")
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg.train.seeds)

    print(f"=== MCI-GRU baseline | market={cfg.market} | smoke={args.smoke} | seeds={seeds} ===")
    init_qlib(cfg)

    print("Preparing data ...")
    prepared = prepare_data(cfg)
    print(f"  universe={len(prepared.universe)} features={prepared.features}")
    for nm in ("train", "valid", "test"):
        s = prepared.splits[nm]
        print(f"  {nm}: days={len(s.dates)} x_ts={s.x_ts.shape} x_graph={s.x_graph.shape}")

    print("Building graph ...")
    graph = build_or_load_graph(cfg, prepared.universe)
    print(f"  edges={int(graph[0].size(1))}")

    device = resolve_device(cfg)
    print(f"  device={device}")

    all_metrics = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pred_df, train_info = train_one_seed(cfg, prepared, graph, seed, device, seed_dir / "best.pth")
        pred_df.to_parquet(seed_dir / "pred_test.parquet")
        print(f"  predictions: {pred_df.shape} | test_rankic={train_info['test_rankic']:.4f}")

        metrics, report = run_backtest(cfg, pred_df)
        metrics["seed"] = seed
        metrics.update(train_info)
        with open(seed_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        report.to_parquet(seed_dir / "report.parquet")

        if args.smoke:
            _smoke_asserts(cfg, prepared, pred_df, metrics)

        all_metrics.append(metrics)
        print(f"  ARR={metrics['annualized_return']:.4f} "
              f"excess={metrics['excess_annualized_return']:.4f} "
              f"Sharpe(IR)={metrics['information_ratio']:.4f} "
              f"MDD={metrics['max_drawdown']:.4f} "
              f"RankIC={metrics['RankIC']:.4f}")

    agg = summarize(all_metrics)
    pd.DataFrame(all_metrics).to_csv(out / "per_seed.csv", index=False)
    pd.DataFrame(agg).to_csv(out / "summary.csv")
    with open(out / "summary.json", "w") as fh:
        json.dump(agg, fh, indent=2)

    print("\n=== summary (mean +/- std over seeds) ===")
    print(pd.DataFrame(agg).to_string())
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
