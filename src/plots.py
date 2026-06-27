"""Matplotlib visual reports for ratings, popular items, and the alpha sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_alpha_sweep(alpha_sweep: pd.DataFrame, outpath: str | Path) -> None:
    """Plot ranking metrics across hybrid alpha values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ["precision", "recall", "ndcg"]:
        if metric in alpha_sweep.columns:
            ax.plot(alpha_sweep["alpha"], alpha_sweep[metric], marker="o", label=metric)
    ax.set_xlabel("Hybrid alpha: 0 = content, 1 = collaborative")
    ax.set_ylabel("Metric value")
    ax.set_title("Hybrid alpha sweep")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_hist(ratings: pd.DataFrame, outpath: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratings["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating distribution")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def top_popular(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    topn: int = 20,
    outpath: str | Path | None = None,
) -> pd.DataFrame:
    pop = train.groupby("movie_id")["rating"].mean().reset_index(name="avg_rating")
    pop["count"] = train.groupby("movie_id")["rating"].count().values
    pop = (
        pop.merge(movies, on="movie_id")
        .sort_values(["avg_rating", "count"], ascending=False)
        .head(topn)
    )
    if outpath:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(pop["title"][::-1], pop["avg_rating"][::-1])
        ax.set_xlabel("Avg Rating")
        ax.set_title("Top Movies (by avg rating)")
        fig.tight_layout()
        fig.savefig(outpath, dpi=160)
        plt.close(fig)
    return pop
