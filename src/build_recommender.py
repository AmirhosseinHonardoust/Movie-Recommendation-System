import argparse, os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from utils import train_test_split_by_user, build_ui_matrix
from baselines import build_baseline_scores

ALPHA_SWEEP_VALUES = tuple(round(i / 10, 1) for i in range(11))

def build_alpha_sweep(eval_model, alpha_values=ALPHA_SWEEP_VALUES):
    """Evaluate hybrid blend weights and return a tidy metrics table.

    ``alpha=0`` is content-only and ``alpha=1`` is collaborative-only. Values
    between them blend collaborative and content scores.
    """
    rows = []
    for alpha in alpha_values:
        alpha = float(alpha)
        metrics = eval_model(alpha)
        rows.append({
            "alpha": alpha,
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "ndcg": float(metrics.get("ndcg", 0.0)),
        })
    return pd.DataFrame(rows)

def best_alpha_by_metric(alpha_sweep, metric="ndcg"):
    """Return the best alpha row as a plain dictionary."""
    if alpha_sweep.empty:
        return {"alpha": None, "metric": metric, "value": None}
    if metric not in alpha_sweep.columns:
        raise ValueError(f"metric must be one of {list(alpha_sweep.columns)}, got {metric!r}")
    ordered = alpha_sweep.sort_values([metric, "precision", "recall"], ascending=False)
    best = ordered.iloc[0]
    return {
        "alpha": float(best["alpha"]),
        "metric": metric,
        "value": float(best[metric]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "ndcg": float(best["ndcg"]),
    }

def plot_alpha_sweep(alpha_sweep, outpath):
    """Plot ranking metrics across hybrid alpha values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ["precision", "recall", "ndcg"]:
        if metric in alpha_sweep.columns:
            ax.plot(alpha_sweep["alpha"], alpha_sweep[metric], marker="o", label=metric)
    ax.set_xlabel("Hybrid alpha: 0 = content, 1 = collaborative")
    ax.set_ylabel("Metric value")
    ax.set_title("Hybrid alpha sweep")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def ensure_outdir(p): os.makedirs(p, exist_ok=True)

def plot_hist(ratings, outpath):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(ratings["rating"], bins=[0.5,1.5,2.5,3.5,4.5,5.5])
    ax.set_xticks([1,2,3,4,5]); ax.set_xlabel("Rating"); ax.set_ylabel("Count")
    ax.set_title("Rating distribution")
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def top_popular(train, movies, topn=20, outpath=None):
    pop = train.groupby("movie_id")["rating"].mean().reset_index(name="avg_rating")
    pop["count"] = train.groupby("movie_id")["rating"].count().values
    pop = pop.merge(movies, on="movie_id").sort_values(["avg_rating","count"], ascending=False).head(topn)
    if outpath:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(pop["title"][::-1], pop["avg_rating"][::-1])
        ax.set_xlabel("Avg Rating"); ax.set_title("Top Movies (by avg rating)")
        fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)
    return pop

def build_content_item_sims(movies):
    tfidf = TfidfVectorizer(token_pattern=r"[^,]+")
    X = tfidf.fit_transform(movies["genres"])
    sims = cosine_similarity(X)
    return sims, tfidf

def collaborative_scores(train_ui, n_components=50, seed=42):
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

def recommend_for_user(uid, seen_items, collab_row, content_row, alpha=0.6, topk=10):
    n_items = collab_row.shape[0]
    scores = alpha * collab_row
    if content_row is not None:
        scores = scores + (1 - alpha) * content_row
    scores = scores.copy()
    scores[list(seen_items)] = -1e9
    top_idx = np.argpartition(scores, -topk)[-topk:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]

def split_genres(value):
    """Split a comma-separated genre string into clean labels."""
    if pd.isna(value):
        return []
    return [genre.strip() for genre in str(value).split(",") if genre.strip()]

def liked_genre_profile(user_liked_movie_ids, movies):
    """Return liked genres ordered by frequency, then alphabetically."""
    if not user_liked_movie_ids:
        return []

    liked_movie_ids = {int(movie_id) for movie_id in user_liked_movie_ids}
    genre_counts = {}
    for genres in movies.loc[movies["movie_id"].isin(liked_movie_ids), "genres"]:
        for genre in split_genres(genres):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    return [
        genre
        for genre, _ in sorted(
            genre_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

def recommendation_reason(movie_genres, liked_genres):
    """Create a short explanation for a recommended movie."""
    movie_genres = split_genres(movie_genres)
    liked = set(liked_genres)
    overlap = [genre for genre in movie_genres if genre in liked]

    if overlap:
        shown = ", ".join(overlap[:3])
        return f"Shares liked genre signals: {shown}."
    if movie_genres:
        shown = ", ".join(movie_genres[:3])
        return f"High hybrid score among unseen movies; genres: {shown}."
    return "High hybrid score among movies the user has not rated."

def build_recommendation_table(uid, top_idx, scores, movies, liked_movie_ids=None):
    """Build a structured recommendation table for one user.

    ``top_idx`` uses zero-based matrix indices; ``movie_id`` is one-based in the
    synthetic dataset. The returned table is easier to inspect than raw title +
    score lines because it includes rank, movie IDs, genres, and a short reason.
    """
    liked_movie_ids = liked_movie_ids or set()
    liked_genres = liked_genre_profile(liked_movie_ids, movies)
    movie_lookup = movies.set_index("movie_id")

    rows = []
    for rank, (item_idx, score) in enumerate(zip(top_idx, scores), start=1):
        movie_id = int(item_idx) + 1
        movie = movie_lookup.loc[movie_id]
        rows.append({
            "rank": rank,
            "user_id": int(uid),
            "movie_id": movie_id,
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "score": round(float(score), 6),
            "reason": recommendation_reason(movie["genres"], liked_genres),
        })
    return pd.DataFrame(rows)

def write_recommendation_outputs(recs, outdir, uid):
    """Write both structured CSV and readable text recommendation outputs."""
    csv_path = os.path.join(outdir, f"recs_user_{uid}.csv")
    txt_path = os.path.join(outdir, f"recs_user_{uid}.txt")

    recs.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("rank	movie_id	title	genres	score	reason\n")
        for row in recs.itertuples(index=False):
            f.write(
                f"{row.rank}	{row.movie_id}	{row.title}	{row.genres}	"
                f"{float(row.score):.4f}	{row.reason}\n"
            )
    return {"csv": csv_path, "txt": txt_path}

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
    ap.add_argument("--skip-alpha-sweep", action="store_true", help="Skip alpha-sweep CSV/plot generation.")
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
    n_comp = max(2, min(50, min(train_ui.shape)-1))
    collab = collaborative_scores(train_ui, n_components=n_comp, seed=args.seed)

    seen_by_user = {uid-1: set((grp["movie_id"].values - 1)) for uid, grp in train.groupby("user_id")}
    liked_by_user = {uid-1: set((grp.loc[grp["rating"]>=4, "movie_id"].values - 1)) for uid, grp in train.groupby("user_id")}

    truth = {}
    for uid, grp in test.groupby("user_id"):
        rel = set((grp.loc[grp["rating"]>=4, "movie_id"].values - 1))
        if len(rel) > 0:
            truth[uid-1] = rel

    def aggregate_ranking_metrics(recommendations_by_user):
        precs, recs, ndcgs = [], [], []
        for u, top_idx in recommendations_by_user():
            relevant = truth.get(u, set())
            if len(relevant) == 0:
                continue
            hits = sum(1 for it in top_idx if it in relevant)
            precs.append(hits / args.k)
            recs.append(hits / len(relevant))
            ndcgs.append(ndcg_at_k(top_idx, relevant, args.k))
        return {
            "precision": float(np.mean(precs)) if precs else 0.0,
            "recall": float(np.mean(recs)) if recs else 0.0,
            "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        }

    def eval_model(alpha_use):
        def recommendations_by_user():
            for u in range(n_users):
                collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
                liked = liked_by_user.get(u, set())
                content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
                seen = seen_by_user.get(u, set())
                top_idx, _ = recommend_for_user(u, seen, collab_row, content_row, alpha=alpha_use, topk=args.k)
                yield u, top_idx
        return aggregate_ranking_metrics(recommendations_by_user)

    def eval_static_scores(score_vector):
        def recommendations_by_user():
            for u in range(n_users):
                seen = seen_by_user.get(u, set())
                top_idx, _ = recommend_for_user(
                    uid=u,
                    seen_items=seen,
                    collab_row=score_vector,
                    content_row=None,
                    alpha=1.0,
                    topk=args.k,
                )
                yield u, top_idx
        return aggregate_ranking_metrics(recommendations_by_user)

    model_metrics = {
        "collaborative": eval_model(alpha_use=1.0),
        "content": eval_model(alpha_use=0.0),
        "hybrid": eval_model(alpha_use=args.alpha),
    }
    baseline_scores = build_baseline_scores(train, n_items=n_items, seed=args.seed)
    baseline_metrics = {
        name: eval_static_scores(scores) for name, scores in baseline_scores.items()
    }

    alpha_sweep = pd.DataFrame()
    best_alpha = {"alpha": args.alpha, "metric": "ndcg", "value": model_metrics["hybrid"]["ndcg"], **model_metrics["hybrid"]}
    if not args.skip_alpha_sweep:
        alpha_sweep = build_alpha_sweep(eval_model, ALPHA_SWEEP_VALUES)
        alpha_sweep.to_csv(os.path.join(args.outdir, "alpha_sweep.csv"), index=False)
        plot_alpha_sweep(alpha_sweep, os.path.join(args.outdir, "alpha_sweep.png"))
        best_alpha = best_alpha_by_metric(alpha_sweep, metric="ndcg")

    comparison_rows = []
    for model_name, values in {**model_metrics, **baseline_metrics}.items():
        comparison_rows.append({"model": model_name, **values})
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["ndcg", "precision", "recall"], ascending=False
    )
    comparison.to_csv(os.path.join(args.outdir, "baseline_comparison.csv"), index=False)

    metrics = {
        "k": args.k,
        "alpha": args.alpha,
        **model_metrics,
        "baselines": baseline_metrics,
        "best_model_by_ndcg": str(comparison.iloc[0]["model"]) if not comparison.empty else None,
        "alpha_sweep": {
            "values": list(ALPHA_SWEEP_VALUES),
            "best_by_ndcg": best_alpha,
        },
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    for uid in [1, 2, 3]:
        u = uid - 1
        collab_row = collab[u] if u < collab.shape[0] else np.zeros(n_items)
        liked = liked_by_user.get(u, set())
        content_row = item_sims[list(liked)].mean(axis=0) if len(liked) > 0 else None
        seen = seen_by_user.get(u, set())
        top_idx, scores = recommend_for_user(u, seen, collab_row, content_row, alpha=args.alpha, topk=args.k)
        recs = build_recommendation_table(
            uid=uid,
            top_idx=top_idx,
            scores=scores,
            movies=movies,
            liked_movie_ids={movie_id + 1 for movie_id in liked},
        )
        write_recommendation_outputs(recs, args.outdir, uid)

    print(f"[OK] Finished. Metrics saved to {os.path.join(args.outdir, 'metrics.json')}")

if __name__ == "__main__":
    main()
