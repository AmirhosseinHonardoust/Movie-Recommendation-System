"""Convert a MovieLens dataset into this project's CSV schema.

MovieLens (e.g. ``ml-latest-small``) ships ``movies.csv`` with pipe-separated
genres and sparse, non-contiguous IDs, and ``ratings.csv`` with half-star
ratings and timestamps. This project expects contiguous one-based IDs,
comma-separated genres, and integer ratings from 1 to 5, so the loader remaps
IDs, rewrites genres, and rounds ratings.

The transformation in ``convert_movielens`` is pure and unit-tested. Fetching
the dataset over the network is left as a manual step (see the README); this
module only converts files that already exist locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert_movielens(
    ml_movies: pd.DataFrame, ml_ratings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(movies, ratings)`` frames in this project's schema.

    Parameters
    ----------
    ml_movies:
        MovieLens movies frame with ``movieId``, ``title``, ``genres`` columns.
    ml_ratings:
        MovieLens ratings frame with ``userId``, ``movieId``, ``rating`` columns.
    """
    for frame, required, name in (
        (ml_movies, {"movieId", "title", "genres"}, "movies"),
        (ml_ratings, {"userId", "movieId", "rating"}, "ratings"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"MovieLens {name} is missing columns: {missing}")

    # Contiguous one-based IDs, ordered by the original MovieLens IDs.
    movie_id_map = {old: new for new, old in enumerate(sorted(ml_movies["movieId"]), start=1)}
    rated_users = sorted(ml_ratings["userId"].unique())
    user_id_map = {old: new for new, old in enumerate(rated_users, start=1)}

    movies = pd.DataFrame(
        {
            "movie_id": ml_movies["movieId"].map(movie_id_map),
            "title": ml_movies["title"].astype(str),
            "genres": ml_movies["genres"].astype(str).map(_normalize_genres),
        }
    ).sort_values("movie_id")

    keep = ml_ratings[ml_ratings["movieId"].isin(movie_id_map)].copy()
    ratings = pd.DataFrame(
        {
            "user_id": keep["userId"].map(user_id_map),
            "movie_id": keep["movieId"].map(movie_id_map),
            "rating": np.clip(keep["rating"].astype(float).round().astype(int), 1, 5),
        }
    ).sort_values(["user_id", "movie_id"])

    movies = movies.reset_index(drop=True)
    ratings = ratings.drop_duplicates(["user_id", "movie_id"]).reset_index(drop=True)
    return movies, ratings


def _normalize_genres(value: str) -> str:
    """Turn pipe-separated MovieLens genres into a comma-separated string."""
    if not value or value == "(no genres listed)":
        return ""
    parts = [part.strip() for part in value.replace("|", ",").split(",")]
    return ",".join(part for part in parts if part)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MovieLens CSVs to the project schema.")
    ap.add_argument(
        "--indir", required=True, help="Directory with MovieLens movies.csv/ratings.csv."
    )
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    indir = Path(args.indir)
    ml_movies = pd.read_csv(indir / "movies.csv")
    ml_ratings = pd.read_csv(indir / "ratings.csv")
    movies, ratings = convert_movielens(ml_movies, ml_ratings)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    movies.to_csv(outdir / "movies.csv", index=False)
    ratings.to_csv(outdir / "ratings.csv", index=False)
    print(f"[OK] wrote {len(movies)} movies and {len(ratings)} ratings to {outdir.as_posix()}")


if __name__ == "__main__":
    main()
