"""Ranking metrics and hybrid alpha-sweep evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import numpy as np
import pandas as pd

ALPHA_SWEEP_VALUES = tuple(round(i / 10, 1) for i in range(11))


def build_alpha_sweep(
    eval_model: Callable[[float], dict[str, float]],
    alpha_values: Sequence[float] = ALPHA_SWEEP_VALUES,
) -> pd.DataFrame:
    """Evaluate hybrid blend weights and return a tidy metrics table.

    ``alpha=0`` is content-only and ``alpha=1`` is collaborative-only. Values
    between them blend collaborative and content scores.
    """
    rows = []
    for alpha in alpha_values:
        alpha = float(alpha)
        metrics = eval_model(alpha)
        rows.append(
            {
                "alpha": alpha,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "ndcg": float(metrics.get("ndcg", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def best_alpha_by_metric(
    alpha_sweep: pd.DataFrame, metric: str = "ndcg"
) -> dict[str, float | str | None]:
    """Return the best alpha row as a plain dictionary."""
    if alpha_sweep.empty:
        return {"alpha": None, "metric": metric, "value": None}
    if metric not in alpha_sweep.columns:
        raise ValueError(f"metric must be one of {list(alpha_sweep.columns)}, got {metric!r}")
    ordered = alpha_sweep.sort_values([metric, "precision", "recall"], ascending=False)
    best = ordered.iloc[0]
    return {
        "alpha": float(best["alpha"]),
        "metric": metric,
        "value": float(best[metric]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "ndcg": float(best["ndcg"]),
    }


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg if idcg > 0 else 0.0
