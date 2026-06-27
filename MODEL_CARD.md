# Model Card: Movie Recommendation System

This model card summarizes the recommender workflow in this repository. It
follows the spirit of the model-card framework: intended use, data, evaluation,
and limitations. This is a **portfolio and research demo**, not a production
recommender.

## Overview

- **Task:** top-K movie recommendation (ranking).
- **Approaches compared:** content-based (TF-IDF over genres), collaborative
  (TruncatedSVD reconstruction), and a hybrid blend controlled by `alpha`.
- **Baselines:** random, most-popular, average-rating, Bayesian-average, and
  positive-count.

## Intended use

- Learning and demonstrating recommender-system workflow design.
- Comparing models against simple baselines with honest ranking metrics.
- A starting point for experimentation on real interaction data.

### Out of scope

- Production personalization, real user targeting, or ranking decisions.
- Any high-stakes or commercial deployment without further validation,
  monitoring, privacy review, and online experimentation.

## Data

- **Default:** deterministic synthetic movies and ratings from
  `data/generate_ratings.py` (configurable users, movies, density, seed).
- **Optional real data:** MovieLens can be converted to this project's schema
  with `data/load_movielens.py` (contiguous IDs, comma-separated genres,
  integer ratings). The dataset must be downloaded separately.
- **Schema:** `movies(movie_id, title, genres)` and
  `ratings(user_id, movie_id, rating)` with one-based IDs.

## Evaluation

- **Split:** per-user hold-out of up to five ratings for testing.
- **Relevance:** a held-out movie with rating >= 4.
- **Metrics:** Precision@K, Recall@K, NDCG@K, MAP@K, and MRR@K, plus a
  bootstrap utility for confidence intervals. NDCG@K is the headline metric.
- **Reproducibility:** deterministic for a fixed seed and environment. Exact
  numbers track the dependency versions pinned in `requirements.txt`.

On the included synthetic dataset, simple popularity/prior baselines are
competitive with (and sometimes beat) the learned models on NDCG@K. This is
reported honestly rather than hidden; it reflects the synthetic data, not a
claim about real-world quality.

## Limitations and ethical considerations

- Synthetic data does not represent real user behavior; metrics do not transfer
  to real-world recommendation quality.
- No fairness, privacy, or safety review has been performed.
- Genre-only content features are coarse; explanations are heuristic
  genre-overlap summaries, not causal reasons.
- No cold-start handling, online learning, drift monitoring, or A/B testing.

## Maintenance

- Automated unit tests and an end-to-end pipeline test run in CI, along with
  ruff, black, and mypy checks. See `README.md` for the developer workflow.
