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

from models import collaborative_scores, fit_collaborative_model


class CollaborativeScoringTests(unittest.TestCase):
    def _ui(self) -> csr_matrix:
        return csr_matrix(
            np.array(
                [
                    [5.0, 4.0, 0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 5.0, 4.0, 0.0],
                    [0.0, 1.0, 4.0, 5.0, 0.0],
                ]
            )
        )

    def test_scores_match_model_reconstruction(self) -> None:
        train_ui = self._ui()
        actual = collaborative_scores(train_ui, n_components=2, seed=7)
        model = fit_collaborative_model(train_ui, n_components=2, seed=7)

        self.assertEqual(actual.shape, train_ui.shape)
        np.testing.assert_allclose(actual, model.reconstruct())

    def test_scores_use_single_not_double_singular_scaling(self) -> None:
        train_ui = self._ui()
        model = fit_collaborative_model(train_ui, n_components=2, seed=7)

        # Reconstruction multiplies the user factors by components once.
        expected = model.user_factors @ model.components + model.user_means[:, None]
        np.testing.assert_allclose(model.reconstruct(), expected)

        # Scaling again by the singular values would inflate the scores.
        svd = TruncatedSVD(n_components=2, random_state=7)
        svd.fit(train_ui)
        double_scaled = (model.user_factors * svd.singular_values_) @ model.components
        self.assertGreater(
            np.max(np.abs(double_scaled - (expected - model.user_means[:, None]))), 1.0
        )

    def test_mean_centering_changes_scores(self) -> None:
        train_ui = self._ui()
        centered = collaborative_scores(train_ui, n_components=2, seed=7)

        svd = TruncatedSVD(n_components=2, random_state=7)
        raw = svd.inverse_transform(svd.fit_transform(train_ui))
        # Centering by per-user means makes the reconstruction differ from the
        # raw (uncentered) SVD reconstruction.
        self.assertGreater(np.max(np.abs(centered - raw)), 1e-6)

    def test_scores_are_finite(self) -> None:
        scores = collaborative_scores(
            csr_matrix(
                np.array([[5.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 2.0], [1.0, 0.0, 5.0, 0.0]])
            ),
            n_components=2,
            seed=42,
        )
        self.assertEqual(scores.shape, (3, 4))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
