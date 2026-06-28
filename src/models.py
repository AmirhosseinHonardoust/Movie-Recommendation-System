"""Recommendation scoring models: content, collaborative, and ranking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_item_sims(movies: pd.DataFrame) -> tuple[np.ndarray, TfidfVectorizer]:
    tfidf = TfidfVectorizer(token_pattern=r"[^,]+")
    X = tfidf.fit_transform(movies["genres"])
    sims = cosine_similarity(X)
    return sims, tfidf


@dataclass
class CollaborativeModel:
    """A fitted collaborative model that reconstructs user-item scores.

    The model stores the SVD user factors and item components plus the per-user
    mean rating used for centering. ``reconstruct`` returns the dense score
    matrix; because the factors and means are kept, this is exactly the matrix
    produced at fit time and can be persisted and reloaded without retraining.
    """

    user_factors: np.ndarray  # (n_users, n_components)
    components: np.ndarray  # (n_components, n_items)
    user_means: np.ndarray  # (n_users,)

    def reconstruct(self) -> np.ndarray:
        scores = self.user_factors @ self.components
        return np.asarray(scores + self.user_means[:, None], dtype=float)


def fit_collaborative_model(
    train_ui: csr_matrix, n_components: int = 50, seed: int = 42
) -> CollaborativeModel:
    """Fit a mean-centered truncated-SVD collaborative model.

    Each user's observed ratings are centered by that user's mean before the
    SVD so the factors capture preference deviations rather than overall rating
    level; the mean is added back during reconstruction. Centering measurably
    improves ranking quality over factorizing the raw rating matrix.

    ``TruncatedSVD.fit_transform`` already returns the user embeddings in the
    reduced latent space, so reconstruction multiplies them by ``components_``
    once (via ``inverse_transform``); scaling again by ``singular_values_``
    would double-count them and inflate scores.
    """
    dense = train_ui.toarray()
    observed = dense > 0
    counts = observed.sum(axis=1)
    user_means = np.where(counts > 0, dense.sum(axis=1) / np.maximum(counts, 1), 0.0)
    centered = np.where(observed, dense - user_means[:, None], 0.0)

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(csr_matrix(centered))
    return CollaborativeModel(
        user_factors=np.asarray(user_factors, dtype=float),
        components=np.asarray(svd.components_, dtype=float),
        user_means=np.asarray(user_means, dtype=float),
    )


def collaborative_scores(
    train_ui: csr_matrix, n_components: int = 50, seed: int = 42
) -> np.ndarray:
    """Return reconstructed user-item scores from the collaborative model."""
    return fit_collaborative_model(train_ui, n_components=n_components, seed=seed).reconstruct()


def recommend_for_user(
    uid: int,
    seen_items: Iterable[int],
    collab_row: np.ndarray,
    content_row: np.ndarray | None,
    alpha: float = 0.6,
    topk: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    scores = alpha * collab_row
    if content_row is not None:
        scores = scores + (1 - alpha) * content_row
    scores = scores.copy()
    scores[list(seen_items)] = -1e9
    top_idx = np.argpartition(scores, -topk)[-topk:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]
