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

from build_recommender import (
    ALPHA_SWEEP_VALUES,
    best_alpha_by_metric,
    build_alpha_sweep,
    plot_alpha_sweep,
)


class AlphaSweepTests(unittest.TestCase):
    def test_default_alpha_values_cover_content_to_collaborative(self) -> None:
        self.assertEqual(ALPHA_SWEEP_VALUES[0], 0.0)
        self.assertEqual(ALPHA_SWEEP_VALUES[-1], 1.0)
        self.assertEqual(len(ALPHA_SWEEP_VALUES), 11)
        self.assertTrue(np.allclose(np.diff(ALPHA_SWEEP_VALUES), 0.1))

    def test_build_alpha_sweep_returns_tidy_metric_rows(self) -> None:
        def fake_eval(alpha: float) -> dict[str, float]:
            return {
                "precision": alpha / 10,
                "recall": 1 - alpha / 10,
                "ndcg": 1 - abs(alpha - 0.4),
            }

        sweep = build_alpha_sweep(fake_eval, alpha_values=[0.0, 0.4, 1.0])

        self.assertEqual(list(sweep.columns), ["alpha", "precision", "recall", "ndcg"])
        self.assertEqual(sweep.shape[0], 3)
        self.assertEqual(sweep["alpha"].tolist(), [0.0, 0.4, 1.0])
        for metric in ["precision", "recall", "ndcg"]:
            self.assertTrue(((sweep[metric] >= 0) & (sweep[metric] <= 1)).all())

    def test_best_alpha_by_metric_selects_highest_ndcg(self) -> None:
        sweep = pd.DataFrame(
            [
                {"alpha": 0.0, "precision": 0.1, "recall": 0.1, "ndcg": 0.2},
                {"alpha": 0.5, "precision": 0.2, "recall": 0.2, "ndcg": 0.7},
                {"alpha": 1.0, "precision": 0.3, "recall": 0.3, "ndcg": 0.4},
            ]
        )

        best = best_alpha_by_metric(sweep, metric="ndcg")

        self.assertEqual(best["alpha"], 0.5)
        self.assertEqual(best["metric"], "ndcg")
        self.assertAlmostEqual(best["value"], 0.7)

    def test_plot_alpha_sweep_writes_file(self) -> None:
        sweep = pd.DataFrame(
            [
                {"alpha": 0.0, "precision": 0.1, "recall": 0.2, "ndcg": 0.3},
                {"alpha": 1.0, "precision": 0.2, "recall": 0.1, "ndcg": 0.4},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            outpath = Path(tmp) / "alpha_sweep.png"
            plot_alpha_sweep(sweep, outpath)
            self.assertTrue(outpath.exists())
            self.assertGreater(outpath.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
