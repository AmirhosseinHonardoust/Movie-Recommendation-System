# Movie Recommendation System
Movie recommendation system with Python. Implements content-based filtering (TF-IDF + cosine similarity), collaborative filtering with matrix factorization (TruncatedSVD), and a hybrid approach. Evaluates with Precision@K, Recall@K, and NDCG. Includes rating distribution plots, top movies, and sample recommendations.

---

## Features
- Synthetic ratings dataset with users, movies, and multi-label genres
- Content-based: TF‑IDF on genres + item–item cosine similarity
- Collaborative: user–item matrix factorization (TruncatedSVD)
<<<<<<< HEAD
- KNN: item-based nearest neighbors recommendation
- Hybrid: weighted combination of content, collaborative, and KNN scores
=======
- Hybrid: weighted combination of content & collaborative scores
>>>>>>> 81fb6d98d82d9e37ed32e8b71c93cb78a568f321
- Evaluation: Precision@K, Recall@K, NDCG@K (per-user & macro)
- Visuals: rating distribution, top movies, sample recommendations
- Outputs: metrics JSON, recommendation files, and charts

---

## Project Structure
```
movie-recommendation-system/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/
│  └─ generate_ratings.py
├─ src/
│  ├─ build_recommender.py
│  └─ utils.py
└─ outputs/
   └─ figures & reports (auto-created)
```

---

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Generate Synthetic Dataset
```bash
python data/generate_ratings.py --users 800 --movies 1200 --seed 42 --outdir data
# Produces: data/ratings.csv, data/movies.csv
```

---

## Train, Evaluate & Recommend
```bash
<<<<<<< HEAD
python src/build_recommender.py --ratings data/ratings.csv --movies data/movies.csv --outdir outputs --k 10 --alpha 0.6 --knn_weight 0.3 --knn_neighbors 20 --seed 42
```

## Streamlit Web App
```bash
python app.py
```

**Outputs**
- `outputs/metrics.json` – Precision@K, Recall@K, NDCG@K (content, collab, hybrid, knn_hybrid)
=======
python src/build_recommender.py --ratings data/ratings.csv --movies data/movies.csv --outdir outputs --k 10 --alpha 0.6 --seed 42
```

**Outputs**
- `outputs/metrics.json` – Precision@K, Recall@K, NDCG@K (content, collab, hybrid)
>>>>>>> 81fb6d98d82d9e37ed32e8b71c93cb78a568f321
- `outputs/recs_user_*.txt` – sample recommendations for a few users
- `outputs/ratings_hist.png`
- `outputs/top_movies.png`

---

## Example Results (K=10, α=0.6)

### Evaluation Metrics
| Method           | Precision@10 | Recall@10 | NDCG@10 |
|------------------|--------------|-----------|---------|
| Collaborative    | 0.0024       | 0.0107    | 0.0059  |
| Content-based    | 0.0026       | 0.0129    | 0.0073  |
| Hybrid (α=0.6)   | 0.0024       | 0.0107    | 0.0060  |

➡️ On this sparse synthetic dataset, **content-based** slightly outperforms collaborative and hybrid.

---

### Ratings Distribution
<img width="960" height="640" alt="ratings_hist" src="https://github.com/user-attachments/assets/c498d323-2659-43b7-b9d3-1a5f2ee517b9" />

---

### Top Movies by Avg Rating
<img width="1280" height="960" alt="top_movies" src="https://github.com/user-attachments/assets/525ee9d8-70c5-48e6-a721-18481d64123a" />

---

### Sample Recommendations

**User 1**
```
Movie 959    41.08
Movie 541    36.28
Movie 952    34.83
Movie 95     34.81
Movie 40     33.30
```

**User 2**
```
Movie 915    47.02
Movie 122    38.27
Movie 77     38.18
Movie 1097   37.65
Movie 952    37.04
```

**User 3**
```
Movie 800    52.00
Movie 410    45.77
Movie 9      45.59
Movie 738    41.34
Movie 713    41.00
```

---

## Notes
- Positive items are defined as ratings ≥ 4 for evaluation.
- The hybrid weight `alpha` controls emphasis on collaborative (α→1) vs. content (α→0).
