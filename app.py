from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_recommender import build_content_item_sims, collaborative_scores
from utils import build_item_knn_model, build_ui_matrix, knn_item_scores


def _ensure_streamlit_import():
    try:
        import streamlit as st  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit==1.37.1"])
    import streamlit as st  # type: ignore
    return st


st = _ensure_streamlit_import()


def _should_bootstrap():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is None
    except Exception:
        return False


def _launch_streamlit():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--global.developmentMode=false"]
    raise SystemExit(stcli.main())


if __name__ == "__main__" and _should_bootstrap():
    _launch_streamlit()


st.set_page_config(
    page_title="Movie Recommendation Studio",
    page_icon="movie",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(59, 130, 246, 0.18), transparent 26%),
            radial-gradient(circle at 90% 8%, rgba(14, 165, 233, 0.16), transparent 24%),
            linear-gradient(180deg, #08111f 0%, #0f172a 55%, #111827 100%);
        color: #f8fafc;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
        color: #f8fafc;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(3, 7, 18, 0.96), rgba(30, 41, 59, 0.96));
        color: white;
        box-shadow: 0 24px 60px rgba(2, 6, 23, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
    }
    .eyebrow {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(255, 255, 255, 0.72);
        margin-bottom: 0.55rem;
    }
    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 3.2vw, 3.1rem);
        line-height: 1.05;
        letter-spacing: -0.05em;
    }
    .hero p {
        margin: 0.8rem 0 0;
        max-width: 70ch;
        color: rgba(255, 255, 255, 0.80);
        font-size: 1rem;
    }
    .panel {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 1rem 1rem 0.85rem;
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.20);
        margin-bottom: 1rem;
    }
    .panel p {
        margin: 0;
        color: rgba(248, 250, 252, 0.84);
        line-height: 1.55;
    }
    .section-title {
        font-size: 1.03rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.65rem;
        letter-spacing: -0.02em;
    }
    .form-wrap {
        background: rgba(15, 23, 42, 0.94);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1rem 1rem 0.5rem;
        box-shadow: 0 14px 28px rgba(2, 6, 23, 0.22);
        margin-bottom: 1rem;
    }
    div[data-testid="stForm"] label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    div[data-testid="stForm"] input,
    div[data-testid="stForm"] textarea,
    div[data-testid="stForm"] [role="combobox"] {
        background: #0f172a !important;
        color: #ffffff !important;
    }
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .movie-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: 0 10px 22px rgba(2, 6, 23, 0.22);
        margin-bottom: 0.85rem;
    }
    .movie-title {
        font-size: 1.14rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.2rem 0;
    }
    .movie-meta {
        color: rgba(226, 232, 240, 0.80);
        font-size: 0.92rem;
        margin: 0 0 0.65rem 0;
    }
    .score-pill {
        display: inline-block;
        padding: 0.28rem 0.68rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.16);
        color: #dbeafe;
        font-size: 0.84rem;
        font-weight: 800;
    }
    .stat-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.10);
        height: 100%;
    }
    .stat-label {
        color: rgba(248, 250, 252, 0.72);
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .results-header {
        margin: 0.25rem 0 0.25rem;
        color: #ffffff;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: -0.03em;
    }
    .results-subtitle {
        margin: 0 0 0.85rem 0;
        color: rgba(248, 250, 252, 0.76);
        line-height: 1.5;
    }
    .name-bar {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.18);
        margin-bottom: 0.85rem;
        color: rgba(248, 250, 252, 0.92);
    }
    .name-bar strong {
        color: #ffffff;
    }
    .info-pill {
        display: inline-block;
        padding: 0.26rem 0.6rem;
        border-radius: 999px;
        margin: 0.18rem 0.35rem 0 0;
        background: rgba(255,255,255,0.07);
        color: rgba(248,250,252,0.90);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.82rem;
    }
    .info-panel {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 1rem;
    }
    .stDataFrame, .stDataFrame * {
        color: #0f172a !important;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    ratings = pd.read_csv(ROOT / "data" / "ratings.csv")
    movies = pd.read_csv(ROOT / "data" / "movies.csv")
    return ratings, movies


@st.cache_resource
def build_models(ratings: pd.DataFrame, movies: pd.DataFrame, knn_neighbors: int, seed: int):
    n_users = int(ratings["user_id"].nunique())
    n_items = int(movies["movie_id"].nunique())
    ui = build_ui_matrix(ratings, n_users, n_items)
    content_sims, _ = build_content_item_sims(movies)
    n_comp = max(2, min(50, min(ui.shape) - 1))
    collab = collaborative_scores(ui, n_components=n_comp, seed=seed)
    knn_model, item_matrix = build_item_knn_model(ui, n_neighbors=knn_neighbors)
    return content_sims, collab, knn_model, item_matrix


def score_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    content_sims,
    collab,
    knn_model,
    item_matrix,
    knn_neighbors: int,
    alpha: float,
    knn_weight: float,
    model_name: str,
):
    n_items = len(movies)
    user_ratings = ratings[ratings["user_id"] == user_id]
    seen = set((user_ratings["movie_id"].values - 1).tolist())
    liked = set((user_ratings.loc[user_ratings["rating"] >= 4, "movie_id"].values - 1).tolist())
    rating_map = {int(mid) - 1: float(rating) for mid, rating in zip(user_ratings["movie_id"].values, user_ratings["rating"].values)}

    user_idx = user_id - 1
    collab_row = collab[user_idx] if user_idx < collab.shape[0] else np.zeros(n_items)
    content_row = content_sims[list(liked)].mean(axis=0) if liked else None
    knn_row = knn_item_scores(seen, rating_map, knn_model, item_matrix, n_items, n_neighbors=knn_neighbors)

    if model_name == "Collaborative":
        scores = collab_row.copy()
    elif model_name == "Content-based":
        scores = content_row.copy() if content_row is not None else np.zeros(n_items)
    elif model_name == "KNN":
        scores = knn_row.copy()
    else:
        scores = alpha * collab_row
        if content_row is not None:
            scores += (1 - alpha) * content_row
        scores += knn_weight * knn_row

    scores = np.array(scores, dtype=float)
    scores[list(seen)] = -np.inf
    return scores, seen, liked, content_row, knn_row


def top_genres_for_user(liked_ids: set[int], movies: pd.DataFrame) -> str:
    if not liked_ids:
        return "No rated favorites yet"
    genre_series = movies.iloc[list(liked_ids)]["genres"].dropna().astype(str)
    genres = (
        genre_series.str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(3)
        .index
        .tolist()
    )
    return ", ".join(genres) if genres else "No genre signal yet"


def recommendation_explanation(model_name: str, alpha: float, knn_weight: float) -> str:
    if model_name == "Collaborative":
        return "This mode looks at users with similar taste and recommends movies they also liked."
    if model_name == "Content-based":
        return "This mode recommends movies that share genres with the titles the user liked."
    if model_name == "KNN":
        return "This mode finds movies nearest to the titles the user already watched."
    return f"Hybrid mode combines collaborative ({alpha:.2f}), content ({1 - alpha:.2f}), and KNN ({knn_weight:.2f}) signals."


def signal_tags(model_name: str) -> list[str]:
    if model_name == "Collaborative":
        return ["Similar users", "Seen history", "Collaborative patterns"]
    if model_name == "Content-based":
        return ["Liked genres", "Genre overlap", "Movie metadata"]
    if model_name == "KNN":
        return ["Similar movies", "Nearest neighbors", "Item proximity"]
    return ["Hybrid score", "User taste", "Movie similarity"]


def render_movie_card(row: pd.Series, score: float):
    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{row["title"]}</div>
            <div class="movie-meta">{row.get("genres", "Unknown genre")}</div>
            <span class="score-pill">Score {score:.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


ratings, movies = load_data()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Movie recommendation engine</div>
        <h1>Find the next film faster.</h1>
        <p>
            A cleaner interface for hybrid recommendations powered by content-based, collaborative,
            and KNN signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="panel">
        <div class="section-title">How this app works</div>
        <p>
            Choose a user, pick a recommendation model, then press <strong>Ara</strong>.
            The app ranks movies the user has not seen yet and shows the strongest matches first.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "search_ready" not in st.session_state:
    st.session_state.search_ready = False
if "form_user_id" not in st.session_state:
    st.session_state.form_user_id = 1
if "form_model_name" not in st.session_state:
    st.session_state.form_model_name = "Hybrid"
if "form_top_n" not in st.session_state:
    st.session_state.form_top_n = 8
if "form_alpha" not in st.session_state:
    st.session_state.form_alpha = 0.6
if "form_knn_weight" not in st.session_state:
    st.session_state.form_knn_weight = 0.3
if "form_knn_neighbors" not in st.session_state:
    st.session_state.form_knn_neighbors = 20
if "form_seed" not in st.session_state:
    st.session_state.form_seed = 42

st.markdown(
    """
    <div class="panel">
        <div class="section-title">Search form</div>
        <p>Fill the form, choose a model, and press <strong>Ara</strong> to generate recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("recommendation_form", clear_on_submit=False):
    left_form, right_form = st.columns([1.1, 0.9], gap="large")

    with left_form:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=int(ratings["user_id"].max()),
            value=int(st.session_state.form_user_id),
            step=1,
        )
        model_name = st.selectbox(
            "Recommendation model",
            ["Hybrid", "Collaborative", "Content-based", "KNN"],
            index=["Hybrid", "Collaborative", "Content-based", "KNN"].index(st.session_state.form_model_name),
        )
        top_n = st.slider("How many recommendations?", 5, 20, int(st.session_state.form_top_n))
        seed = st.number_input("Seed", value=int(st.session_state.form_seed), step=1)

    with right_form:
        alpha = st.slider("Collaborative weight", 0.0, 1.0, float(st.session_state.form_alpha), 0.05)
        knn_weight = st.slider("KNN weight", 0.0, 1.0, float(st.session_state.form_knn_weight), 0.05)
        knn_neighbors = st.slider("KNN neighbors", 5, 50, int(st.session_state.form_knn_neighbors), 1)
        st.markdown(
            '<div class="info-panel">Hybrid mode uses all three signals. The weights matter only when Hybrid is selected.</div>',
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button("Ara", use_container_width=True)

if submitted:
    st.session_state.search_ready = True
    st.session_state.form_user_id = int(user_id)
    st.session_state.form_model_name = model_name
    st.session_state.form_top_n = int(top_n)
    st.session_state.form_alpha = float(alpha)
    st.session_state.form_knn_weight = float(knn_weight)
    st.session_state.form_knn_neighbors = int(knn_neighbors)
    st.session_state.form_seed = int(seed)

if not st.session_state.search_ready:
    st.info("Fill the form and press Ara to see recommendations.")
    st.stop()

user_id = int(st.session_state.form_user_id)
model_name = st.session_state.form_model_name
top_n = int(st.session_state.form_top_n)
alpha = float(st.session_state.form_alpha)
knn_weight = float(st.session_state.form_knn_weight)
knn_neighbors = int(st.session_state.form_knn_neighbors)
seed = int(st.session_state.form_seed)

content_sims, collab, knn_model, item_matrix = build_models(
    ratings,
    movies,
    knn_neighbors=knn_neighbors,
    seed=seed,
)

scores, seen, liked, content_row, knn_row = score_user(
    user_id=user_id,
    ratings=ratings,
    movies=movies,
    content_sims=content_sims,
    collab=collab,
    knn_model=knn_model,
    item_matrix=item_matrix,
    knn_neighbors=knn_neighbors,
    alpha=alpha,
    knn_weight=knn_weight,
    model_name=model_name,
)

top_idx = np.argpartition(scores, -top_n)[-top_n:]
top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
recommended = movies.iloc[top_idx].copy()
recommended["score"] = scores[top_idx]

avg_rating = float(ratings["rating"].mean())
top_movie_row = ratings.groupby("movie_id")["rating"].mean().sort_values(ascending=False).index[0]
top_movie_title = movies.loc[movies["movie_id"] == top_movie_row, "title"].iloc[0]
top_genres = top_genres_for_user(liked, movies)
why_text = recommendation_explanation(model_name, float(alpha), float(knn_weight))
tags = signal_tags(model_name)

st.markdown(
    f"""
    <div class="panel">
        <div class="section-title">How recommendations are chosen</div>
        <p style="margin-bottom: 0.7rem;">{why_text}</p>
        <div>
            {''.join(f'<span class="info-pill">{tag}</span>' for tag in tags)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_a, stat_b, stat_c = st.columns(3)
with stat_a:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Total movies</div>
            <p class="stat-value">{len(movies):,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat_b:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Average rating</div>
            <p class="stat-value">{avg_rating:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat_c:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Top rated title</div>
            <p class="stat-value">{top_movie_title}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_reco, tab_data = st.tabs(["Recommendations", "Explore data"])

with tab_reco:
    left, right = st.columns([1.4, 0.9], gap="large")

    with left:
        st.markdown('<div class="results-header">Recommended for you</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="results-subtitle">Below are the unseen movies ranked by the selected model.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="name-bar"><strong>Selected user:</strong> {user_id} &nbsp; | &nbsp; <strong>Model:</strong> {model_name} &nbsp; | &nbsp; <strong>Top N:</strong> {top_n}</div>',
            unsafe_allow_html=True,
        )
        rec_left, rec_right = st.columns(2, gap="large")
        for idx, (_, row) in enumerate(recommended.iterrows()):
            with rec_left if idx % 2 == 0 else rec_right:
                render_movie_card(row, float(row["score"]))

        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Why these titles?</div>
                <p>
                    The app scores unseen movies using the selected model and removes anything the user already watched.
                    The highest-scoring movies are shown first.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Profile summary</div>', unsafe_allow_html=True)
        st.markdown(f"**Selected user:** {user_id}")
        st.markdown(f"**Seen movies:** {len(seen)}")
        st.markdown(f"**Liked movies:** {len(liked)}")
        st.markdown(f"**Primary taste:** {top_genres}")
        st.markdown(f"**Model:** {model_name}")
        st.markdown(f"**KNN neighbors:** {knn_neighbors}")
        st.markdown(f"**Recommendation logic:** {why_text}")
        st.caption("Already seen movies are filtered out. Only unseen titles are ranked.")
        st.progress(min(len(seen) / max(len(movies), 1), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Liked movies</div>', unsafe_allow_html=True)
        liked_titles = movies.loc[movies["movie_id"].isin([i + 1 for i in sorted(liked)])].head(8)
        if liked_titles.empty:
            st.info("No liked movies found for this user yet.")
        else:
            st.dataframe(liked_titles[["title", "genres"]], use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    data_left, data_right = st.columns([1, 1], gap="large")

    with data_left:
        st.markdown('<div class="section-title">Most popular movies</div>', unsafe_allow_html=True)
        popular = (
            ratings.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .merge(movies, on="movie_id")
            .sort_values(["mean", "count"], ascending=False)
            .head(12)
        )
        st.dataframe(popular[["title", "genres", "mean", "count"]], use_container_width=True, hide_index=True)

    with data_right:
        st.markdown('<div class="section-title">Rating distribution</div>', unsafe_allow_html=True)
        hist = np.histogram(ratings["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0]
        st.bar_chart(pd.DataFrame({"rating": [1, 2, 3, 4, 5], "count": hist}).set_index("rating"))

        st.markdown('<div class="section-title" style="margin-top: 1rem;">Genre counts</div>', unsafe_allow_html=True)
        genre_counts = (
            movies["genres"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
        )
        st.bar_chart(genre_counts)
