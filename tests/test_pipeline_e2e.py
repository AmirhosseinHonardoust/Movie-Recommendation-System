from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from generate_ratings import make_movies, make_ratings


class PipelineEndToEndTests(unittest.TestCase):
    """Run the full build_recommender workflow as a subprocess on tiny data."""

    def test_workflow_produces_all_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            out_dir = tmpdir / "outputs"
            data_dir.mkdir()

            movies = make_movies(n_movies=40, seed=7)
            ratings = make_ratings(n_users=40, movies=movies, density=0.4, seed=7)
            movies.to_csv(data_dir / "movies.csv", index=False)
            ratings.to_csv(data_dir / "ratings.csv", index=False)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "build_recommender.py"),
                    "--ratings",
                    str(data_dir / "ratings.csv"),
                    "--movies",
                    str(data_dir / "movies.csv"),
                    "--outdir",
                    str(out_dir),
                    "--k",
                    "10",
                    "--alpha",
                    "0.6",
                    "--seed",
                    "7",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            expected = [
                "metrics.json",
                "baseline_comparison.csv",
                "alpha_sweep.csv",
                "alpha_sweep.png",
                "ratings_hist.png",
                "top_movies.png",
                "recs_user_1.csv",
                "recs_user_2.csv",
                "recs_user_3.csv",
                "recs_user_1.txt",
                "recs_user_2.txt",
                "recs_user_3.txt",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), f"missing output: {name}")

            metrics = json.loads((out_dir / "metrics.json").read_text())
            for model in ("collaborative", "content", "hybrid"):
                for metric in ("precision", "recall", "ndcg", "map", "mrr"):
                    value = metrics[model][metric]
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(len(metrics["baselines"]), 5)

            for uid in (1, 2, 3):
                recs = pd.read_csv(out_dir / f"recs_user_{uid}.csv")
                self.assertEqual(len(recs), 10)
                self.assertEqual(recs["rank"].tolist(), list(range(1, 11)))
                self.assertTrue((recs["score"].values[:-1] >= recs["score"].values[1:]).all())


if __name__ == "__main__":
    unittest.main()
