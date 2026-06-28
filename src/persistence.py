"""Train and persist a recommender model as portable JSON (no pickle).

The artifact stores the fitted collaborative factors (user factors, item
components, and per-user means) together with the parameters needed to serve
recommendations. JSON is used deliberately instead of pickle: it is
human-inspectable, safe to load from an untrusted source, and decouples the
saved model from the exact Python object layout. Content similarity is a
deterministic function of the movie metadata and is recomputed at serve time,
so it is not stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models import CollaborativeModel, fit_collaborative_model
from utils import build_ui_matrix

MODEL_FORMAT = "movie-recommender-model"
MODEL_VERSION = 1


def choose_n_components(n_users: int, n_items: int) -> int:
    """Pick a safe SVD rank for the given matrix shape."""
    return max(2, min(50, min(n_users, n_items) - 1))


def train_recommender(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    alpha: float = 0.6,
    k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the collaborative model on all ratings and return a JSON-ready dict."""
    n_users = int(ratings["user_id"].max())
    n_items = int(movies["movie_id"].nunique())
    n_components = choose_n_components(n_users, n_items)

    ui = build_ui_matrix(ratings, n_users, n_items)
    model = fit_collaborative_model(ui, n_components=n_components, seed=seed)

    return {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "params": {
            "n_components": n_components,
            "alpha": float(alpha),
            "k": int(k),
            "seed": int(seed),
        },
        "n_users": n_users,
        "n_items": n_items,
        "collaborative": {
            "user_factors": model.user_factors.tolist(),
            "components": model.components.tolist(),
            "user_means": model.user_means.tolist(),
        },
    }


def save_model(model_dict: dict[str, Any], path: str | Path) -> None:
    """Write a model dict to ``path`` as JSON."""
    Path(path).write_text(json.dumps(model_dict))


def load_model(path: str | Path) -> dict[str, Any]:
    """Load and validate a model dict from a JSON file."""
    data: dict[str, Any] = json.loads(Path(path).read_text())
    if data.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not a {MODEL_FORMAT} file")
    if data.get("version") != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {data.get('version')}")
    return data


def collaborative_model_from_dict(model_dict: dict[str, Any]) -> CollaborativeModel:
    """Rebuild a :class:`CollaborativeModel` from a loaded model dict."""
    collab = model_dict["collaborative"]
    return CollaborativeModel(
        user_factors=np.asarray(collab["user_factors"], dtype=float),
        components=np.asarray(collab["components"], dtype=float),
        user_means=np.asarray(collab["user_means"], dtype=float),
    )
