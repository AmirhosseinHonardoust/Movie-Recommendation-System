from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from load_movielens import convert_movielens


class MovieLensLoaderTests(unittest.TestCase):
    def _ml_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ml_movies = pd.DataFrame(
            {
                "movieId": [10, 50, 200],
                "title": ["A (1995)", "B (2001)", "C (2010)"],
                "genres": ["Action|Sci-Fi", "Comedy", "(no genres listed)"],
            }
        )
        ml_ratings = pd.DataFrame(
            {
                "userId": [5, 5, 9, 9],
                "movieId": [10, 50, 200, 999],
                "rating": [4.5, 2.0, 3.5, 5.0],
                "timestamp": [1, 2, 3, 4],
            }
        )
        return ml_movies, ml_ratings

    def test_ids_are_remapped_to_contiguous_one_based(self) -> None:
        movies, ratings = convert_movielens(*self._ml_frames())

        self.assertEqual(movies["movie_id"].tolist(), [1, 2, 3])
        self.assertEqual(sorted(ratings["user_id"].unique()), [1, 2])
        self.assertTrue(ratings["movie_id"].between(1, 3).all())

    def test_genres_and_ratings_are_normalized(self) -> None:
        movies, ratings = convert_movielens(*self._ml_frames())

        self.assertEqual(movies.loc[movies["movie_id"] == 1, "genres"].iloc[0], "Action,Sci-Fi")
        self.assertEqual(movies.loc[movies["movie_id"] == 3, "genres"].iloc[0], "")
        self.assertTrue(ratings["rating"].between(1, 5).all())
        self.assertEqual(ratings["rating"].dtype.kind, "i")

    def test_ratings_for_unknown_movies_are_dropped(self) -> None:
        _, ratings = convert_movielens(*self._ml_frames())
        # movieId 999 has no movie row and must be dropped.
        self.assertEqual(len(ratings), 3)

    def test_missing_columns_raise(self) -> None:
        with self.assertRaises(ValueError):
            convert_movielens(pd.DataFrame({"movieId": [1]}), pd.DataFrame({"userId": [1]}))


if __name__ == "__main__":
    unittest.main()
