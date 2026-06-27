"""Generate top-K recommendations for a single user (inference path).

Unlike ``build_recommender.py``, which holds out a test set for evaluation,
this command trains on all available ratings and returns recommendations for a
requested ``--user``. It reuses the same scoring models so results are
consistent with the evaluated pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from io_utils import ensure_outdir, write_recommendation_outputs
from models import build_content_item_sims, collaborative_scores, recommend_for_user
from reporting import build_recommendation_table
from utils import build_ui_matrix


def recommend_for_user_id(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    k: int = 10,
    alpha: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a top-``k`` recommendation table for ``user_id``.

    Movies the user has already rated are excluded. Liked movies (rating >= 4)
    drive the content signal and the per-row explanation.
    """
    if user_id not in set(ratings["user_id"].unique()):
        raise ValueError(f"user_id {user_id} is not present in the ratings")

    n_users = int(ratings["user_id"].max())
    n_items = movies["movie_id"].nunique()

    ui = build_ui_matrix(ratings, n_users, n_items)
    item_sims, _ = build_content_item_sims(movies)
    n_comp = max(2, min(50, min(ui.shape) - 1))
    collab = collaborative_scores(ui, n_components=n_comp, seed=seed)

    u = user_id - 1
    user_ratings = ratings[ratings["user_id"] == user_id]
    seen = set(user_ratings["movie_id"].to_numpy() - 1)
    liked = set(user_ratings.loc[user_ratings["rating"] >= 4, "movie_id"].to_numpy() - 1)

    collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
    content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
    top_idx, scores = recommend_for_user(u, seen, collab_row, content_row, alpha=alpha, topk=k)

    return build_recommendation_table(
        uid=user_id,
        top_idx=top_idx,
        scores=scores,
        movies=movies,
        liked_movie_ids={movie_id + 1 for movie_id in liked},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend movies for a single user.")
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--movies", required=True)
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None, help="If set, also write CSV/TXT outputs here.")
    args = ap.parse_args()

    ratings = pd.read_csv(args.ratings)
    movies = pd.read_csv(args.movies)
    recs = recommend_for_user_id(
        ratings, movies, args.user, k=args.k, alpha=args.alpha, seed=args.seed
    )

    if args.outdir:
        ensure_outdir(args.outdir)
        write_recommendation_outputs(recs, args.outdir, args.user)

    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
