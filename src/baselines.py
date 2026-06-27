"""Baseline recommenders for the movie recommendation workflow.

The baselines in this module are intentionally simple. They provide context for
interpreting collaborative, content-based, and hybrid recommendation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASELINE_NAMES = (
    "random",
    "most_popular",
    "average_rating",
    "bayesian_average",
    "positive_count",
)


@dataclass(frozen=True)
class ItemStatistics:
    """Precomputed item-level statistics used by baseline recommenders."""

    rating_count: np.ndarray
    rating_sum: np.ndarray
    rating_mean: np.ndarray
    positive_count: np.ndarray
    global_mean: float


def compute_item_statistics(
    ratings: pd.DataFrame,
    n_items: int,
    positive_threshold: float = 4.0,
) -> ItemStatistics:
    """Compute dense item statistics from a ratings dataframe.

    Parameters
    ----------
    ratings:
        Ratings dataframe with ``movie_id`` and ``rating`` columns. Movie IDs are
        expected to be one-indexed, matching the rest of the project.
    n_items:
        Number of movies/items in the catalog.
    positive_threshold:
        Rating value used to count positive interactions.
    """
    required = {"movie_id", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
    if n_items <= 0:
        raise ValueError("n_items must be positive")

    rating_count = np.zeros(n_items, dtype=float)
    rating_sum = np.zeros(n_items, dtype=float)
    positive_count = np.zeros(n_items, dtype=float)

    movie_idx = ratings["movie_id"].to_numpy(dtype=int) - 1
    if len(movie_idx) and (movie_idx.min() < 0 or movie_idx.max() >= n_items):
        raise ValueError("movie_id values must be between 1 and n_items")

    rating_values = ratings["rating"].to_numpy(dtype=float)
    np.add.at(rating_count, movie_idx, 1.0)
    np.add.at(rating_sum, movie_idx, rating_values)
    np.add.at(positive_count, movie_idx, (rating_values >= positive_threshold).astype(float))

    global_mean = float(np.mean(rating_values)) if len(rating_values) else 0.0
    rating_mean = np.divide(
        rating_sum,
        rating_count,
        out=np.full(n_items, global_mean, dtype=float),
        where=rating_count > 0,
    )

    return ItemStatistics(
        rating_count=rating_count,
        rating_sum=rating_sum,
        rating_mean=rating_mean,
        positive_count=positive_count,
        global_mean=global_mean,
    )


def build_baseline_scores(
    train: pd.DataFrame,
    n_items: int,
    seed: int = 42,
    bayesian_prior_weight: float = 20.0,
) -> dict[str, np.ndarray]:
    """Return score vectors for simple recommender baselines.

    Each vector has one score per item. The same vector is used for every user;
    seen items are removed later by the common recommendation function.
    """
    stats = compute_item_statistics(train, n_items=n_items)
    rng = np.random.default_rng(seed)

    bayesian_average = (stats.rating_sum + bayesian_prior_weight * stats.global_mean) / (
        stats.rating_count + bayesian_prior_weight
    )

    return {
        "random": rng.random(n_items),
        "most_popular": stats.rating_count.astype(float),
        "average_rating": stats.rating_mean.astype(float),
        "bayesian_average": bayesian_average.astype(float),
        "positive_count": stats.positive_count.astype(float),
    }
