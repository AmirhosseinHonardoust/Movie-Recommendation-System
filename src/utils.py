import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def train_test_split_by_user(ratings: pd.DataFrame, test_k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for _uid, grp in ratings.groupby("user_id"):
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
