from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(DATA_DIR))

from generate_ratings import GENRES, make_movies, make_ratings


class DataGenerationTests(unittest.TestCase):
    def test_make_movies_has_expected_schema_and_valid_genres(self) -> None:
        movies = make_movies(n_movies=25, seed=123)

        self.assertEqual(list(movies.columns), ["movie_id", "title", "genres"])
        self.assertEqual(len(movies), 25)
        self.assertEqual(movies["movie_id"].tolist(), list(range(1, 26)))
        self.assertTrue(movies["title"].str.startswith("Movie ").all())

        allowed = set(GENRES)
        for value in movies["genres"]:
            genres = set(value.split(","))
            self.assertGreaterEqual(len(genres), 1)
            self.assertLessEqual(len(genres), 3)
            self.assertTrue(genres.issubset(allowed))

    def test_make_movies_is_deterministic_for_same_seed(self) -> None:
        left = make_movies(n_movies=15, seed=42)
        right = make_movies(n_movies=15, seed=42)

        pd.testing.assert_frame_equal(left, right)

    def test_make_ratings_has_valid_schema_and_ranges(self) -> None:
        movies = make_movies(n_movies=40, seed=42)
        ratings = make_ratings(n_users=12, movies=movies, density=0.20, seed=42)

        self.assertEqual(list(ratings.columns), ["user_id", "movie_id", "rating"])
        self.assertFalse(ratings.empty)
        self.assertTrue(ratings["user_id"].between(1, 12).all())
        self.assertTrue(ratings["movie_id"].between(1, 40).all())
        self.assertTrue(ratings["rating"].between(1, 5).all())
        self.assertFalse(ratings.duplicated(["user_id", "movie_id"]).any())


if __name__ == "__main__":
    unittest.main()
