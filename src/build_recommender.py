import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    build_item_knn_model,
    build_ui_matrix,
    knn_item_scores,
    train_test_split_by_user,
)


def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)


def plot_hist(ratings, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratings["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating distribution")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def top_popular(train, movies, topn=20, outpath=None):
    pop = train.groupby("movie_id")["rating"].mean().reset_index(name="avg_rating")
    pop["count"] = train.groupby("movie_id")["rating"].count().values
    pop = pop.merge(movies, on="movie_id").sort_values(["avg_rating", "count"], ascending=False).head(topn)
    if outpath:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(pop["title"][::-1], pop["avg_rating"][::-1])
        ax.set_xlabel("Avg Rating")
        ax.set_title("Top Movies (by avg rating)")
        fig.tight_layout()
        fig.savefig(outpath, dpi=160)
        plt.close(fig)
    return pop


def build_content_item_sims(movies):
    tfidf = TfidfVectorizer(token_pattern=r"[^,]+")
    x = tfidf.fit_transform(movies["genres"])
    sims = cosine_similarity(x)
    return sims, tfidf


def collaborative_scores(train_ui, n_components=50, seed=42):
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    u = svd.fit_transform(train_ui)
    s = svd.singular_values_
    vt = svd.components_
    scores = (u * s) @ vt
    return scores


def recommend_for_user(uid, seen_items, collab_row, content_row, knn_row=None, alpha=0.6, topk=10, knn_weight=0.3):
    scores = alpha * collab_row
    if content_row is not None:
        scores = scores + (1 - alpha) * content_row
    if knn_row is not None:
        scores = scores + knn_weight * knn_row
    scores = scores.copy()
    scores[list(seen_items)] = -1e9
    top_idx = np.argpartition(scores, -topk)[-topk:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]


def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--movies", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--knn_weight", type=float, default=0.3)
    ap.add_argument("--knn_neighbors", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    ratings = pd.read_csv(args.ratings)
    movies = pd.read_csv(args.movies)

    n_users = ratings["user_id"].nunique()
    n_items = movies["movie_id"].nunique()

    plot_hist(ratings, os.path.join(args.outdir, "ratings_hist.png"))
    top_popular(ratings, movies, outpath=os.path.join(args.outdir, "top_movies.png"))

    train, test = train_test_split_by_user(ratings, test_k=5, seed=args.seed)
    train_ui = build_ui_matrix(train, n_users, n_items)

    item_sims, tfidf = build_content_item_sims(movies)
    n_comp = max(2, min(50, min(train_ui.shape) - 1))
    collab = collaborative_scores(train_ui, n_components=n_comp, seed=args.seed)
    knn_model, item_matrix = build_item_knn_model(train_ui, n_neighbors=args.knn_neighbors)

    seen_by_user = {uid - 1: set((grp["movie_id"].values - 1)) for uid, grp in train.groupby("user_id")}
    liked_by_user = {uid - 1: set((grp.loc[grp["rating"] >= 4, "movie_id"].values - 1)) for uid, grp in train.groupby("user_id")}
    rating_map_by_user = {
        uid - 1: {int(mid) - 1: float(r) for mid, r in zip(grp["movie_id"].values, grp["rating"].values)}
        for uid, grp in train.groupby("user_id")
    }

    truth = {}
    for uid, grp in test.groupby("user_id"):
        rel = set((grp.loc[grp["rating"] >= 4, "movie_id"].values - 1))
        if len(rel) > 0:
            truth[uid - 1] = rel

    def eval_model(alpha_use, include_knn=False):
        precs, recs, ndcgs = [], [], []
        for u in range(n_users):
            collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
            liked = liked_by_user.get(u, set())
            content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
            seen = seen_by_user.get(u, set())
            knn_row = None
            if include_knn:
                knn_row = knn_item_scores(
                    seen,
                    rating_map_by_user.get(u, {}),
                    knn_model,
                    item_matrix,
                    n_items,
                    n_neighbors=args.knn_neighbors,
                )
            top_idx, _ = recommend_for_user(
                u,
                seen,
                collab_row,
                content_row,
                knn_row=knn_row,
                alpha=alpha_use,
                topk=args.k,
                knn_weight=args.knn_weight,
            )
            relevant = truth.get(u, set())
            if len(relevant) == 0:
                continue
            hits = sum(1 for it in top_idx if it in relevant)
            precs.append(hits / args.k)
            recs.append(hits / len(relevant))
            ndcgs.append(ndcg_at_k(top_idx, relevant, args.k))
        return (
            float(np.mean(precs)) if precs else 0.0,
            float(np.mean(recs)) if recs else 0.0,
            float(np.mean(ndcgs)) if ndcgs else 0.0,
        )

    p_c, r_c, n_c = eval_model(alpha_use=1.0)
    p_t, r_t, n_t = eval_model(alpha_use=0.0)
    p_h, r_h, n_h = eval_model(alpha_use=args.alpha)
    p_k, r_k, n_k = eval_model(alpha_use=args.alpha, include_knn=True)

    metrics = {
        "k": args.k,
        "alpha": args.alpha,
        "knn_weight": args.knn_weight,
        "knn_neighbors": args.knn_neighbors,
        "collaborative": {"precision": p_c, "recall": r_c, "ndcg": n_c},
        "content": {"precision": p_t, "recall": r_t, "ndcg": n_t},
        "hybrid": {"precision": p_h, "recall": r_h, "ndcg": n_h},
        "knn_hybrid": {"precision": p_k, "recall": r_k, "ndcg": n_k},
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for uid in [1, 2, 3]:
        u = uid - 1
        collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
        liked = liked_by_user.get(u, set())
        content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
        seen = seen_by_user.get(u, set())
        knn_row = knn_item_scores(
            seen,
            rating_map_by_user.get(u, {}),
            knn_model,
            item_matrix,
            n_items,
            n_neighbors=args.knn_neighbors,
        )
        top_idx, scores = recommend_for_user(
            u,
            seen,
            collab_row,
            content_row,
            knn_row=knn_row,
            alpha=args.alpha,
            topk=args.k,
            knn_weight=args.knn_weight,
        )
        titles = movies.set_index("movie_id").loc[top_idx + 1, "title"].tolist()
        with open(os.path.join(args.outdir, f"recs_user_{uid}.txt"), "w", encoding="utf-8") as f:
            for t, s in zip(titles, scores):
                f.write(f"{t}\t{float(s):.4f}\n")

    print("[OK] Finished. Metrics saved to outputs/metrics.json")


if __name__ == "__main__":
    main()
