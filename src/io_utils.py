"""Filesystem helpers for the recommender workflow outputs."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def ensure_outdir(p: str | Path) -> None:
    os.makedirs(p, exist_ok=True)


def write_recommendation_outputs(
    recs: pd.DataFrame, outdir: str | Path, uid: int
) -> dict[str, str]:
    """Write both structured CSV and readable text recommendation outputs."""
    csv_path = os.path.join(outdir, f"recs_user_{uid}.csv")
    txt_path = os.path.join(outdir, f"recs_user_{uid}.txt")

    recs.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("rank\tmovie_id\ttitle\tgenres\tscore\treason\n")
        for row in recs.itertuples(index=False):
            f.write(
                f"{row.rank}\t{row.movie_id}\t{row.title}\t{row.genres}\t"
                f"{float(row.score):.4f}\t{row.reason}\n"
            )
    return {"csv": csv_path, "txt": txt_path}
