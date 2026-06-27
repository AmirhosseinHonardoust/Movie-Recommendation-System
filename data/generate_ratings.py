from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _density(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("density must be in the interval (0, 1]")
    return parsed


def make_movies(n_movies: int, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic synthetic movie catalog.

    Each movie receives one to three genres sampled from a fixed genre list. The
    genre labels within each movie are sorted so output is stable and easier to
    diff across runs.
    """
    if n_movies <= 0:
        raise ValueError("n_movies must be positive")

    rng = np.random.default_rng(seed)
    rows: list[list[object]] = []

    for movie_id in range(1, n_movies + 1):
        n_genres = int(rng.integers(1, 4))
        genres = sorted(rng.choice(GENRES, size=n_genres, replace=False).tolist())
        rows.append([movie_id, f"Movie {movie_id:04d}", ",".join(genres)])

    return pd.DataFrame(rows, columns=["movie_id", "title", "genres"])


def _validate_movies(movies: pd.DataFrame) -> None:
    required = {"movie_id", "title", "genres"}
    missing = sorted(required - set(movies.columns))
    if missing:
        raise ValueError(f"movies is missing required columns: {missing}")
    if movies.empty:
        raise ValueError("movies must contain at least one row")
    if movies["movie_id"].duplicated().any():
        raise ValueError("movie_id values must be unique")
    if movies["genres"].isna().any():
        raise ValueError("genres must not contain missing values")


def _all_genres_from_movies(movies: pd.DataFrame) -> list[str]:
    """Return a deterministic sorted list of genres present in the catalog."""
    genre_values: set[str] = set()
    for value in movies["genres"].astype(str):
        for genre in value.split(","):
            cleaned = genre.strip()
            if cleaned:
                genre_values.add(cleaned)
    if not genre_values:
        raise ValueError("at least one genre must be present")
    return sorted(genre_values)


def make_ratings(
    n_users: int,
    movies: pd.DataFrame,
    density: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic synthetic user/movie ratings.

    The generator combines latent user/item factors with genre-level item
    structure. It is intentionally synthetic and meant for demo workflows, not
    for benchmarking real-world recommender quality.
    """
    if n_users <= 0:
        raise ValueError("n_users must be positive")
    if not 0 < density <= 1:
        raise ValueError("density must be in the interval (0, 1]")
    _validate_movies(movies)

    rng = np.random.default_rng(seed)
    n_items = len(movies)
    latent_dim = 8

    user_factors = rng.normal(0, 1, size=(n_users, latent_dim))
    item_factors = rng.normal(0, 1, size=(n_items, latent_dim))

    # Important for reproducibility: iterate through genres in sorted order.
    # A plain set has process-dependent ordering and can change generated
    # ratings even with the same random seed.
    genre_to_vec = {
        genre: rng.normal(0, 0.6, size=latent_dim) for genre in _all_genres_from_movies(movies)
    }

    item_bias = rng.normal(0, 0.3, size=n_items)
    for row_index, genre_text in enumerate(movies["genres"].astype(str)):
        genres = [genre.strip() for genre in genre_text.split(",") if genre.strip()]
        genre_vector = np.mean([genre_to_vec[genre] for genre in genres], axis=0)
        item_factors[row_index] += genre_vector * 0.7

    rows: list[list[int]] = []
    base_ratings_per_user = max(5, int(round(density * n_items)))

    for user_id in range(1, n_users + 1):
        jitter = int(rng.integers(-10, 10))
        n_ratings = min(n_items, max(5, base_ratings_per_user + jitter))
        item_indices = rng.choice(n_items, size=n_ratings, replace=False)
        user_vector = user_factors[user_id - 1]

        for item_index in item_indices:
            raw_score = (
                float(user_vector @ item_factors[item_index])
                + float(item_bias[item_index])
                + float(rng.normal(0, 1.0))
            )
            positive_probability = 1 / (1 + np.exp(-raw_score / 2))
            rating = int(np.clip(np.rint(positive_probability * 4), 0, 4) + 1)
            rows.append(
                [
                    user_id,
                    int(movies["movie_id"].iloc[item_index]),
                    rating,
                ]
            )

    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])
    ratings = ratings.sort_values(["user_id", "movie_id"]).reset_index(drop=True)
    return ratings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic movie and ratings CSV files."
    )
    parser.add_argument("--users", type=_positive_int, default=800)
    parser.add_argument("--movies", type=_positive_int, default=1200)
    parser.add_argument("--density", type=_density, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    movies = make_movies(n_movies=args.movies, seed=args.seed)
    ratings = make_ratings(
        n_users=args.users,
        movies=movies,
        density=args.density,
        seed=args.seed,
    )

    movies_path = outdir / "movies.csv"
    ratings_path = outdir / "ratings.csv"
    movies.to_csv(movies_path, index=False)
    ratings.to_csv(ratings_path, index=False)

    print(f"[OK] wrote {len(movies)} movies to {movies_path.as_posix()}")
    print(f"[OK] wrote {len(ratings)} ratings to {ratings_path.as_posix()}")


if __name__ == "__main__":
    main()
