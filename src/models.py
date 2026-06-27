"""Recommendation scoring models: content, collaborative, and ranking."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


def build_content_item_sims(movies: pd.DataFrame) -> tuple[np.ndarray, TfidfVectorizer]:
    tfidf = TfidfVectorizer(token_pattern=r"[^,]+")
    X = tfidf.fit_transform(movies["genres"])
    sims = cosine_similarity(X)
    return sims, tfidf


def collaborative_scores(
    train_ui: csr_matrix, n_components: int = 50, seed: int = 42
) -> np.ndarray:
    """Return reconstructed user-item scores from a truncated SVD model.

    ``TruncatedSVD.fit_transform`` already returns the user embeddings in the
    reduced latent space. Multiplying those embeddings by ``singular_values_``
    again double-counts the singular values and inflates recommendation scores.
    ``inverse_transform`` is the scikit-learn-supported reconstruction path.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(train_ui)
    scores = svd.inverse_transform(user_factors)
    return np.asarray(scores, dtype=float)


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
