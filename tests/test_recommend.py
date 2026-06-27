from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_ratings import make_movies, make_ratings

from recommend import recommend_for_user_id


class RecommendCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = make_movies(n_movies=40, seed=1)
        self.ratings = make_ratings(n_users=20, movies=self.movies, density=0.3, seed=1)

    def test_recommend_returns_valid_ranked_table(self) -> None:
        recs = recommend_for_user_id(self.ratings, self.movies, user_id=1, k=10, seed=1)

        self.assertEqual(
            list(recs.columns),
            ["rank", "user_id", "movie_id", "title", "genres", "score", "reason"],
        )
        self.assertEqual(recs["rank"].tolist(), list(range(1, 11)))
        self.assertTrue((recs["user_id"] == 1).all())
        scores = recs["score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_recommend_excludes_already_seen_movies(self) -> None:
        seen = set(self.ratings.loc[self.ratings["user_id"] == 1, "movie_id"])
        recs = recommend_for_user_id(self.ratings, self.movies, user_id=1, k=10, seed=1)
        self.assertTrue(set(recs["movie_id"]).isdisjoint(seen))

    def test_recommend_rejects_unknown_user(self) -> None:
        with self.assertRaises(ValueError):
            recommend_for_user_id(self.ratings, self.movies, user_id=9999, seed=1)


if __name__ == "__main__":
    unittest.main()
