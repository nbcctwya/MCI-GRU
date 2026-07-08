"""Configuration loading for the MCI-GRU qlib baseline.

A YAML file is loaded into a recursive namespace (attribute access). The
``--smoke`` flag applies a tiny end-to-end override so the whole pipeline can be
exercised in <2 minutes on CPU without touching the real experiment.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Namespace:
    """Recursively exposes a dict via attribute access (``cfg.model.hidden``)."""

    def __init__(self, data: Any) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                self.__dict__[k] = Namespace(v) if isinstance(v, dict) else v
        else:  # pragma: no cover - defensive
            raise TypeError(f"Namespace expects a dict, got {type(data)}")

    def __getattr__(self, name):  # only called when normal lookup fails
        raise AttributeError(name)

    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Namespace) else v
        return out

    def __repr__(self) -> str:
        return f"Namespace({self.to_dict()})"


# Tiny overrides applied on top of the market YAML for an end-to-end smoke run.
SMOKE_OVERRIDES = {
    "universe": {
        "start": "2023-01-01",
        "end": "2023-04-30",
        "presence_threshold": 0.0,
        "max_instruments": 20,
    },
    "splits": {
        "train": {"start": "2023-01-01", "end": "2023-02-28"},
        "valid": {"start": "2023-03-01", "end": "2023-03-31"},
        "test": {"start": "2023-04-01", "end": "2023-04-30"},
    },
    "graph": {"judge_value": 0.3, "corr_window_days": 20},
    "train": {"num_seeds": 1, "seeds": [0], "num_epochs": 2, "device": "cpu"},
    "backtest": {"topk": 5, "n_drop": 1},
    "output_dir": None,  # rewritten below to "<market>_smoke"
}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, smoke: bool = False) -> Namespace:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    if smoke:
        overrides = copy.deepcopy(SMOKE_OVERRIDES)
        overrides["output_dir"] = f"outputs/{cfg['market']}_smoke"
        _deep_update(cfg, overrides)
    else:
        if not cfg["output_dir"].startswith("outputs/"):
            cfg["output_dir"] = f"outputs/{cfg['output_dir']}"

    return Namespace(cfg)
