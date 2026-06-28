from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_ratings import make_movies, make_ratings

from persistence import (
    collaborative_model_from_dict,
    load_model,
    save_model,
    train_recommender,
)
from recommend import recommend_for_user_id


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = make_movies(n_movies=40, seed=3)
        self.ratings = make_ratings(n_users=25, movies=self.movies, density=0.3, seed=3)
        self.model = train_recommender(self.ratings, self.movies, seed=3)

    def test_round_trip_preserves_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            save_model(self.model, path)
            loaded = load_model(path)

        a = collaborative_model_from_dict(self.model)
        b = collaborative_model_from_dict(loaded)
        np.testing.assert_allclose(a.reconstruct(), b.reconstruct())
        self.assertEqual(loaded["params"], self.model["params"])

    def test_loaded_model_matches_fresh_training(self) -> None:
        collab = collaborative_model_from_dict(self.model)
        for uid in (1, 5, 10):
            fresh = recommend_for_user_id(self.ratings, self.movies, uid, seed=3)
            served = recommend_for_user_id(
                self.ratings, self.movies, uid, seed=3, collab_model=collab
            )
            self.assertTrue(fresh.equals(served), f"user {uid} mismatch")

    def test_artifact_is_plain_json_not_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            save_model(self.model, path)
            raw = path.read_bytes()

        # Parses as JSON (would raise if it were pickle or anything else).
        parsed = json.loads(raw.decode("utf-8"))
        self.assertEqual(parsed["format"], "movie-recommender-model")
        # Pickle streams start with the protocol-2 opcode b"\x80"; JSON never does.
        self.assertNotEqual(raw[:1], b"\x80")

    def test_load_rejects_unknown_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"format": "nope", "version": 1}))
            with self.assertRaises(ValueError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()
