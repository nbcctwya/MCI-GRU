"""qlib backtest + metrics for one seed's test predictions.

Uses ``qlib.contrib.evaluate.backtest_daily`` with a ``TopkDropoutStrategy``
and the user's transaction costs (open_cost=buy, close_cost=sell as fractions),
region-aware ``limit_threshold``/benchmark. Reports portfolio risk metrics
(absolute, excess over benchmark, excess net of cost) and signal IC/RankIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.eva.alpha import calc_ic


def _ra(series: pd.Series) -> dict:
    ra = risk_analysis(series, freq="day")
    out = {}
    for k in ["annualized_return", "information_ratio", "max_drawdown"]:
        out[k] = float(np.nan_to_num(ra.loc[k, "risk"]))
    return out


def run_backtest(cfg, pred_df: pd.DataFrame):
    """pred_df: MultiIndex (datetime, instrument) with 'score' and 'label' columns."""
    bk = cfg.backtest
    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": pred_df["score"],
            "topk": int(bk.topk),
            "n_drop": int(bk.n_drop),
        },
    }
    exchange_kwargs = {
        "open_cost": float(bk.open_cost),
        "close_cost": float(bk.close_cost),
        "min_cost": float(bk.min_cost),
        "deal_price": bk.deal_price,
        "limit_threshold": bk.limit_threshold,
    }
    report, _positions = backtest_daily(
        start_time=cfg.splits.test.start,
        end_time=cfg.splits.test.end,
        strategy=strategy,
        account=float(bk.account),
        benchmark=cfg.benchmark,
        exchange_kwargs=exchange_kwargs,
    )

    ret = report["return"]
    bench = report["bench"]
    cost = report["cost"]
    metrics = {
        "annualized_return": _ra(ret)["annualized_return"],
        "excess_annualized_return": _ra(ret - bench)["annualized_return"],
        "excess_net_annualized_return": _ra(ret - bench - cost)["annualized_return"],
        "information_ratio": _ra(ret - bench)["information_ratio"],
        "max_drawdown": _ra(ret)["max_drawdown"],
        "mean_daily_cost": float(np.nan_to_num(cost.mean())),
    }

    ic, ric = calc_ic(pred_df["score"], pred_df["label"])
    metrics["IC"] = float(np.nan_to_num(ic.mean()))
    metrics["RankIC"] = float(np.nan_to_num(ric.mean()))
    metrics["ICIR"] = float(np.nan_to_num(ic.mean() / ic.std())) if ic.std() != 0 else 0.0
    metrics["RankICIR"] = float(np.nan_to_num(ric.mean() / ric.std())) if ric.std() != 0 else 0.0
    return metrics, report
