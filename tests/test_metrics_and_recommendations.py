from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from metrics import ndcg_at_k
from models import build_content_item_sims, recommend_for_user


class MetricsAndRecommendationTests(unittest.TestCase):
    def test_ndcg_at_k_rewards_relevant_items_near_top(self) -> None:
        relevant = {2, 4}

        perfect = ndcg_at_k([2, 4, 1], relevant, k=3)
        partial = ndcg_at_k([1, 2, 3], relevant, k=3)
        miss = ndcg_at_k([0, 1, 3], relevant, k=3)

        self.assertAlmostEqual(perfect, 1.0)
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, perfect)
        self.assertEqual(miss, 0.0)

    def test_recommend_for_user_excludes_seen_items_and_sorts_scores(self) -> None:
        collab = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        content = np.array([0.0, 0.0, 1.0, 0.0, 0.5])
        seen = {1}

        top_idx, scores = recommend_for_user(
            uid=0,
            seen_items=seen,
            collab_row=collab,
            content_row=content,
            alpha=0.5,
            topk=3,
        )

        self.assertNotIn(1, set(top_idx))
        self.assertEqual(len(top_idx), 3)
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))

    def test_build_content_item_sims_returns_square_similarity_matrix(self) -> None:
        movies = pd.DataFrame(
            {
                "movie_id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "genres": ["Action,Sci-Fi", "Action", "Romance"],
            }
        )

        sims, vectorizer = build_content_item_sims(movies)

        self.assertEqual(sims.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(sims), 1.0))
        self.assertGreater(sims[0, 1], sims[0, 2])
        self.assertGreater(len(vectorizer.get_feature_names_out()), 0)


if __name__ == "__main__":
    unittest.main()
