import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def train_test_split_by_user(ratings: pd.DataFrame, test_k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for uid, grp in ratings.groupby("user_id"):
        idx = np.arange(len(grp))
        if len(idx) <= test_k:
            train_rows.append(grp)
            continue
        test_sel = rng.choice(idx, size=test_k, replace=False)
        mask = np.zeros(len(grp), dtype=bool)
        mask[test_sel] = True
        test_rows.append(grp[mask])
        train_rows.append(grp[~mask])
    train = pd.concat(train_rows, ignore_index=True)
    test = (
        pd.concat(test_rows, ignore_index=True)
        if test_rows
        else pd.DataFrame(columns=ratings.columns)
    )
    return train, test


def build_ui_matrix(ratings: pd.DataFrame, n_users: int, n_items: int):
    rows = ratings["user_id"].values - 1
    cols = ratings["movie_id"].values - 1
    data = ratings["rating"].values.astype(float)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def build_item_knn_model(train_ui, n_neighbors: int = 20, metric: str = "cosine"):
    """Fit an item-based KNN model on the item-user matrix."""
    item_matrix = train_ui.T.tocsr()
    n_items = item_matrix.shape[0]
    if n_items == 0:
        raise ValueError("train_ui must contain at least one item")

    n_neighbors = max(1, min(n_neighbors, n_items))
    model = NearestNeighbors(metric=metric, algorithm="brute", n_neighbors=n_neighbors)
    model.fit(item_matrix)
    return model, item_matrix


def knn_item_scores(seen_items, item_ratings, knn_model, item_matrix, n_items: int, n_neighbors: int = 20):
    """Build recommendation scores from the items a user has already rated."""
    scores = np.zeros(n_items, dtype=float)
    if not seen_items:
        return scores

    n_neighbors = max(1, min(n_neighbors, item_matrix.shape[0]))
    seen_set = set(seen_items)

    for item_id in seen_set:
        distances, indices = knn_model.kneighbors(item_matrix[item_id], n_neighbors=n_neighbors)
        distances = distances.ravel()
        indices = indices.ravel()
        similarities = 1.0 - distances if knn_model.metric == "cosine" else 1.0 / (1.0 + distances)
        base_weight = float(item_ratings.get(item_id, 1.0)) if item_ratings is not None else 1.0
        for neighbor_id, similarity in zip(indices, similarities):
            if neighbor_id == item_id:
                continue
            scores[neighbor_id] += similarity * base_weight

    scores[list(seen_set)] = -np.inf
    return scores
