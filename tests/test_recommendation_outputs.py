from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from io_utils import write_recommendation_outputs
from reporting import (
    build_recommendation_table,
    liked_genre_profile,
    recommendation_reason,
)


class RecommendationOutputTests(unittest.TestCase):
    def _movies(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "movie_id": [1, 2, 3, 4],
                "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
                "genres": ["Action,Sci-Fi", "Romance", "Action,Thriller", "Comedy"],
            }
        )

    def test_liked_genre_profile_orders_genres_by_frequency(self) -> None:
        profile = liked_genre_profile({1, 3}, self._movies())

        self.assertEqual(profile[0], "Action")
        self.assertIn("Sci-Fi", profile)
        self.assertIn("Thriller", profile)

    def test_recommendation_reason_mentions_overlapping_genres(self) -> None:
        reason = recommendation_reason("Action,Comedy", ["Action", "Sci-Fi"])

        self.assertIn("Action", reason)
        self.assertIn("liked genre", reason)

    def test_build_recommendation_table_adds_metadata_and_reasons(self) -> None:
        recs = build_recommendation_table(
            uid=7,
            top_idx=np.array([2, 1]),
            scores=np.array([0.9, 0.7]),
            movies=self._movies(),
            liked_movie_ids={1},
        )

        self.assertEqual(
            list(recs.columns),
            ["rank", "user_id", "movie_id", "title", "genres", "score", "reason"],
        )
        self.assertEqual(recs["rank"].tolist(), [1, 2])
        self.assertEqual(recs["user_id"].tolist(), [7, 7])
        self.assertEqual(recs["movie_id"].tolist(), [3, 2])
        self.assertTrue((recs["reason"].str.len() > 0).all())
        self.assertGreaterEqual(recs.loc[0, "score"], recs.loc[1, "score"])

    def test_write_recommendation_outputs_creates_csv_and_text_files(self) -> None:
        recs = build_recommendation_table(
            uid=2,
            top_idx=np.array([0, 3]),
            scores=np.array([0.8, 0.6]),
            movies=self._movies(),
            liked_movie_ids={3},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_recommendation_outputs(recs, tmpdir, uid=2)
            csv_path = Path(paths["csv"])
            txt_path = Path(paths["txt"])

            self.assertTrue(csv_path.exists())
            self.assertTrue(txt_path.exists())

            loaded = pd.read_csv(csv_path)
            self.assertEqual(len(loaded), 2)
            self.assertIn("reason", loaded.columns)
            self.assertIn("rank\tmovie_id\ttitle\tgenres\tscore\treason", txt_path.read_text())


if __name__ == "__main__":
    unittest.main()
