from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import build_ui_matrix, train_test_split_by_user


class SplitAndMatrixTests(unittest.TestCase):
    def _ratings(self) -> pd.DataFrame:
        rows = []
        for user_id in range(1, 5):
            for movie_id in range(1, 9):
                rows.append(
                    {
                        "user_id": user_id,
                        "movie_id": movie_id,
                        "rating": 1 + ((user_id + movie_id) % 5),
                    }
                )
        return pd.DataFrame(rows)

    def test_train_test_split_by_user_holds_out_requested_rows(self) -> None:
        ratings = self._ratings()
        train, test = train_test_split_by_user(ratings, test_k=2, seed=7)

        self.assertEqual(len(train) + len(test), len(ratings))
        self.assertEqual(test.groupby("user_id").size().to_dict(), {1: 2, 2: 2, 3: 2, 4: 2})
        self.assertEqual(train.groupby("user_id").size().to_dict(), {1: 6, 2: 6, 3: 6, 4: 6})

        train_pairs = set(map(tuple, train[["user_id", "movie_id"]].to_numpy()))
        test_pairs = set(map(tuple, test[["user_id", "movie_id"]].to_numpy()))
        self.assertTrue(train_pairs.isdisjoint(test_pairs))

    def test_train_test_split_keeps_short_user_histories_in_train(self) -> None:
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 2, 2, 2, 2],
                "movie_id": [1, 2, 1, 2, 3, 4, 5, 6],
                "rating": [5, 4, 3, 4, 5, 2, 1, 5],
            }
        )
        train, test = train_test_split_by_user(ratings, test_k=3, seed=42)

        self.assertEqual(len(test[test["user_id"] == 1]), 0)
        self.assertEqual(len(train[train["user_id"] == 1]), 2)
        self.assertEqual(len(test[test["user_id"] == 2]), 3)

    def test_build_ui_matrix_has_expected_shape_and_values(self) -> None:
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 3],
                "movie_id": [1, 3, 2, 4],
                "rating": [5, 4, 3, 2],
            }
        )
        matrix = build_ui_matrix(ratings, n_users=3, n_items=4)

        self.assertEqual(matrix.shape, (3, 4))
        dense = matrix.toarray()
        self.assertEqual(dense[0, 0], 5)
        self.assertEqual(dense[0, 2], 4)
        self.assertEqual(dense[1, 1], 3)
        self.assertEqual(dense[2, 3], 2)
        self.assertEqual(np.count_nonzero(dense), 4)


if __name__ == "__main__":
    unittest.main()
