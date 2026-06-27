"""Command-line entry point that runs the full recommender workflow.

This module wires together the focused modules (data utils, models, metrics,
reporting, plots, and IO) and writes the evaluation artifacts to ``--outdir``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from baselines import build_baseline_scores
from io_utils import ensure_outdir, write_recommendation_outputs
from metrics import (
    ALPHA_SWEEP_VALUES,
    average_precision_at_k,
    best_alpha_by_metric,
    build_alpha_sweep,
    ndcg_at_k,
    reciprocal_rank_at_k,
)
from models import build_content_item_sims, collaborative_scores, recommend_for_user
from plots import plot_alpha_sweep, plot_hist, top_popular
from reporting import build_recommendation_table
from utils import build_ui_matrix, train_test_split_by_user


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--movies", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument(
        "--skip-alpha-sweep", action="store_true", help="Skip alpha-sweep CSV/plot generation."
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    ratings = pd.read_csv(args.ratings)
    movies = pd.read_csv(args.movies)

    n_users = ratings["user_id"].nunique()
    n_items = movies["movie_id"].nunique()

    plot_hist(ratings, os.path.join(args.outdir, "ratings_hist.png"))
    top_popular(ratings, movies, outpath=os.path.join(args.outdir, "top_movies.png"))

    train, test = train_test_split_by_user(ratings, test_k=5, seed=args.seed)
    train_ui = build_ui_matrix(train, n_users, n_items)

    item_sims, tfidf = build_content_item_sims(movies)
    n_comp = max(2, min(50, min(train_ui.shape) - 1))
    collab = collaborative_scores(train_ui, n_components=n_comp, seed=args.seed)

    seen_by_user = {
        uid - 1: set(grp["movie_id"].values - 1) for uid, grp in train.groupby("user_id")
    }
    liked_by_user = {
        uid - 1: set(grp.loc[grp["rating"] >= 4, "movie_id"].values - 1)
        for uid, grp in train.groupby("user_id")
    }

    truth = {}
    for uid, grp in test.groupby("user_id"):
        rel = set(grp.loc[grp["rating"] >= 4, "movie_id"].values - 1)
        if len(rel) > 0:
            truth[uid - 1] = rel

    def aggregate_ranking_metrics(recommendations_by_user):
        precs, recs, ndcgs, maps, mrrs = [], [], [], [], []
        for u, top_idx in recommendations_by_user():
            relevant = truth.get(u, set())
            if len(relevant) == 0:
                continue
            hits = sum(1 for it in top_idx if it in relevant)
            precs.append(hits / args.k)
            recs.append(hits / len(relevant))
            ndcgs.append(ndcg_at_k(top_idx, relevant, args.k))
            maps.append(average_precision_at_k(top_idx, relevant, args.k))
            mrrs.append(reciprocal_rank_at_k(top_idx, relevant, args.k))
        return {
            "precision": float(np.mean(precs)) if precs else 0.0,
            "recall": float(np.mean(recs)) if recs else 0.0,
            "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "map": float(np.mean(maps)) if maps else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        }

    def eval_model(alpha_use):
        def recommendations_by_user():
            for u in range(n_users):
                collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
                liked = liked_by_user.get(u, set())
                content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
                seen = seen_by_user.get(u, set())
                top_idx, _ = recommend_for_user(
                    u, seen, collab_row, content_row, alpha=alpha_use, topk=args.k
                )
                yield u, top_idx

        return aggregate_ranking_metrics(recommendations_by_user)

    def eval_static_scores(score_vector):
        def recommendations_by_user():
            for u in range(n_users):
                seen = seen_by_user.get(u, set())
                top_idx, _ = recommend_for_user(
                    uid=u,
                    seen_items=seen,
                    collab_row=score_vector,
                    content_row=None,
                    alpha=1.0,
                    topk=args.k,
                )
                yield u, top_idx

        return aggregate_ranking_metrics(recommendations_by_user)

    model_metrics = {
        "collaborative": eval_model(alpha_use=1.0),
        "content": eval_model(alpha_use=0.0),
        "hybrid": eval_model(alpha_use=args.alpha),
    }
    baseline_scores = build_baseline_scores(train, n_items=n_items, seed=args.seed)
    baseline_metrics = {
        name: eval_static_scores(scores) for name, scores in baseline_scores.items()
    }

    alpha_sweep = pd.DataFrame()
    best_alpha = {
        "alpha": args.alpha,
        "metric": "ndcg",
        "value": model_metrics["hybrid"]["ndcg"],
        **model_metrics["hybrid"],
    }
    if not args.skip_alpha_sweep:
        alpha_sweep = build_alpha_sweep(eval_model, ALPHA_SWEEP_VALUES)
        alpha_sweep.to_csv(os.path.join(args.outdir, "alpha_sweep.csv"), index=False)
        plot_alpha_sweep(alpha_sweep, os.path.join(args.outdir, "alpha_sweep.png"))
        best_alpha = best_alpha_by_metric(alpha_sweep, metric="ndcg")

    comparison_rows = []
    for model_name, values in {**model_metrics, **baseline_metrics}.items():
        comparison_rows.append({"model": model_name, **values})
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["ndcg", "precision", "recall"], ascending=False
    )
    comparison.to_csv(os.path.join(args.outdir, "baseline_comparison.csv"), index=False)

    metrics = {
        "k": args.k,
        "alpha": args.alpha,
        **model_metrics,
        "baselines": baseline_metrics,
        "best_model_by_ndcg": str(comparison.iloc[0]["model"]) if not comparison.empty else None,
        "alpha_sweep": {
            "values": list(ALPHA_SWEEP_VALUES),
            "best_by_ndcg": best_alpha,
        },
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    for uid in [1, 2, 3]:
        u = uid - 1
        collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
        liked = liked_by_user.get(u, set())
        content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
        seen = seen_by_user.get(u, set())
        top_idx, scores = recommend_for_user(
            u, seen, collab_row, content_row, alpha=args.alpha, topk=args.k
        )
        recs = build_recommendation_table(
            uid=uid,
            top_idx=top_idx,
            scores=scores,
            movies=movies,
            liked_movie_ids={movie_id + 1 for movie_id in liked},
        )
        write_recommendation_outputs(recs, args.outdir, uid)

    print(f"[OK] Finished. Metrics saved to {os.path.join(args.outdir, 'metrics.json')}")


if __name__ == "__main__":
    main()
