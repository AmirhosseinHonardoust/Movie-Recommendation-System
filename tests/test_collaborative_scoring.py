from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from build_recommender import collaborative_scores


class CollaborativeScoringTests(unittest.TestCase):
    def test_collaborative_scores_use_svd_inverse_transform(self) -> None:
        train_ui = csr_matrix(
            np.array(
                [
                    [5.0, 4.0, 0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0, 4.0, 0.0],
                    [0.0, 1.0, 4.0, 5.0, 0.0],
                ]
            )
        )

        actual = collaborative_scores(train_ui, n_components=2, seed=7)

        svd = TruncatedSVD(n_components=2, random_state=7)
        user_factors = svd.fit_transform(train_ui)
        expected = svd.inverse_transform(user_factors)
        old_double_scaled = (user_factors * svd.singular_values_) @ svd.components_

        self.assertEqual(actual.shape, train_ui.shape)
        np.testing.assert_allclose(actual, expected)
        self.assertGreater(np.max(np.abs(old_double_scaled - expected)), 1.0)

    def test_collaborative_scores_are_finite(self) -> None:
        train_ui = csr_matrix(
            np.array(
                [
                    [5.0, 0.0, 3.0, 0.0],
                    [0.0, 4.0, 0.0, 2.0],
                    [1.0, 0.0, 5.0, 0.0],
                ]
            )
        )

        scores = collaborative_scores(train_ui, n_components=2, seed=42)

        self.assertEqual(scores.shape, (3, 4))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
