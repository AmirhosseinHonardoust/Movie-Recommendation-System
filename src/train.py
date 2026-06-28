"""Train the recommender on all ratings and persist it as JSON.

This is the offline training step of a train/serve split: it fits the
collaborative model once and writes a portable artifact that
``recommend.py --model`` can load to generate recommendations without
retraining.
"""

from __future__ import annotations

import argparse

import pandas as pd

from persistence import save_model, train_recommender


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and save the recommender model.")
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--movies", required=True)
    ap.add_argument("--out", default="outputs/model.json", help="Model artifact path.")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratings = pd.read_csv(args.ratings)
    movies = pd.read_csv(args.movies)
    model = train_recommender(ratings, movies, alpha=args.alpha, k=args.k, seed=args.seed)
    save_model(model, args.out)

    params = model["params"]
    print(
        f"[OK] trained on {model['n_users']} users x {model['n_items']} items "
        f"(rank {params['n_components']}); saved to {args.out}"
    )


if __name__ == "__main__":
    main()
