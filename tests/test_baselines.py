from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from baselines import BASELINE_NAMES, build_baseline_scores, compute_item_statistics


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "movie_id": [1, 2, 1, 3, 1, 4],
                "rating": [5, 4, 3, 5, 5, 2],
            }
        )

    def test_compute_item_statistics_returns_dense_vectors(self) -> None:
        stats = compute_item_statistics(self.ratings, n_items=5)

        self.assertEqual(stats.rating_count.shape, (5,))
        self.assertEqual(stats.rating_count[0], 3.0)
        self.assertEqual(stats.positive_count[0], 2.0)
        self.assertAlmostEqual(stats.rating_mean[0], (5 + 3 + 5) / 3)
        self.assertAlmostEqual(stats.rating_mean[4], stats.global_mean)

    def test_build_baseline_scores_has_expected_models_and_shapes(self) -> None:
        scores = build_baseline_scores(self.ratings, n_items=5, seed=123)

        self.assertEqual(set(scores), set(BASELINE_NAMES))
        for values in scores.values():
            self.assertEqual(values.shape, (5,))
            self.assertTrue(np.all(np.isfinite(values)))

    def test_popularity_baselines_rank_frequently_positive_item_highest(self) -> None:
        scores = build_baseline_scores(self.ratings, n_items=5, seed=123)

        self.assertEqual(int(np.argmax(scores["most_popular"])), 0)
        self.assertEqual(int(np.argmax(scores["positive_count"])), 0)
        self.assertGreater(scores["bayesian_average"][0], scores["bayesian_average"][3])

    def test_random_baseline_is_reproducible_for_seed(self) -> None:
        first = build_baseline_scores(self.ratings, n_items=5, seed=7)["random"]
        second = build_baseline_scores(self.ratings, n_items=5, seed=7)["random"]
        different = build_baseline_scores(self.ratings, n_items=5, seed=8)["random"]

        self.assertTrue(np.allclose(first, second))
        self.assertFalse(np.allclose(first, different))


if __name__ == "__main__":
    unittest.main()
