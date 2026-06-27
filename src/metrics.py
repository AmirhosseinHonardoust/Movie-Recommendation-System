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
                "map": float(metrics.get("map", 0.0)),
                "mrr": float(metrics.get("mrr", 0.0)),
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
    row: dict[str, float | str | None] = {
        "alpha": float(best["alpha"]),
        "metric": metric,
        "value": float(best[metric]),
    }
    for column in ("precision", "recall", "ndcg", "map", "mrr"):
        if column in alpha_sweep.columns:
            row[column] = float(best[column])
    return row


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Average precision at ``k`` for a single ranked list.

    Averaged across users this is the Mean Average Precision (MAP@K). The
    denominator is ``min(k, len(relevant))`` so a perfect ranking scores 1.0.
    """
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    denom = min(k, len(relevant))
    return score / denom if denom > 0 else 0.0


def reciprocal_rank_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Reciprocal rank of the first relevant item in the top ``k``.

    Averaged across users this is the Mean Reciprocal Rank (MRR@K).
    """
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for the mean of ``values``."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(means, (1 - confidence) / 2))
    hi = float(np.quantile(means, 1 - (1 - confidence) / 2))
    return (lo, hi)
