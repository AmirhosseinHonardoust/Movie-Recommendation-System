"""Human-readable recommendation tables and genre-based explanations."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd


def split_genres(value: object) -> list[str]:
    """Split a comma-separated genre string into clean labels."""
    if pd.isna(value):
        return []
    return [genre.strip() for genre in str(value).split(",") if genre.strip()]


def liked_genre_profile(user_liked_movie_ids: Collection[int], movies: pd.DataFrame) -> list[str]:
    """Return liked genres ordered by frequency, then alphabetically."""
    if not user_liked_movie_ids:
        return []

    liked_movie_ids = {int(movie_id) for movie_id in user_liked_movie_ids}
    genre_counts: dict[str, int] = {}
    for genres in movies.loc[movies["movie_id"].isin(liked_movie_ids), "genres"]:
        for genre in split_genres(genres):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    return [
        genre
        for genre, _ in sorted(
            genre_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def recommendation_reason(movie_genres: object, liked_genres: Sequence[str]) -> str:
    """Create a short explanation for a recommended movie."""
    movie_genres = split_genres(movie_genres)
    liked = set(liked_genres)
    overlap = [genre for genre in movie_genres if genre in liked]

    if overlap:
        shown = ", ".join(overlap[:3])
        return f"Shares liked genre signals: {shown}."
    if movie_genres:
        shown = ", ".join(movie_genres[:3])
        return f"High hybrid score among unseen movies; genres: {shown}."
    return "High hybrid score among movies the user has not rated."


def build_recommendation_table(
    uid: int,
    top_idx: np.ndarray,
    scores: np.ndarray,
    movies: pd.DataFrame,
    liked_movie_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Build a structured recommendation table for one user.

    ``top_idx`` uses zero-based matrix indices; ``movie_id`` is one-based in the
    synthetic dataset. The returned table is easier to inspect than raw title +
    score lines because it includes rank, movie IDs, genres, and a short reason.
    """
    liked_movie_ids = liked_movie_ids or set()
    liked_genres = liked_genre_profile(liked_movie_ids, movies)
    movie_lookup = movies.set_index("movie_id")

    rows = []
    for rank, (item_idx, score) in enumerate(zip(top_idx, scores, strict=False), start=1):
        movie_id = int(item_idx) + 1
        movie = movie_lookup.loc[movie_id]
        rows.append(
            {
                "rank": rank,
                "user_id": int(uid),
                "movie_id": movie_id,
                "title": str(movie["title"]),
                "genres": str(movie["genres"]),
                "score": round(float(score), 6),
                "reason": recommendation_reason(movie["genres"], liked_genres),
            }
        )
    return pd.DataFrame(rows)
