""" The Netflix Prize data stores rating dates (not times). Ties on the same day
are sorted by movieId, matching data.py, but this does NOT imply true order.
"""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pandas as pd

from data import (
    MIN_RATINGS_PER_ITEM, MIN_RATINGS_PER_USER, MIN_SPAN_DAYS,
    chronological_split, data_hash,
)


def iter_netflix_ratings(directory):
    """Yield (userId, movieId, rating, timestamp) without loading files."""
    for part in range(1, 5):
        path = directory / f"combined_data_{part}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        print(f"Reading {path.name}...", flush=True)
        movie_id = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    movie_id = int(line[:-1])
                    continue
                if movie_id is None:
                    raise ValueError(f"Rating before movie header in {path}")
                user, rating, date = line.split(",")
                # Netflix dates have one-day resolution; midnight is NOT
                # the actual moment each rating was submitted.
                day = np.datetime64(date, "D").astype("int64")
                yield int(user), movie_id, float(rating), int(day) * 86400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netflix-dir", type=Path,
                         default=Path("data/netflix/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/netflix_dev"))
    parser.add_argument("--sample-mod", type=int, default=100,
                        help="Keep user IDs divisible by this number (100 = ~1%%)")
    args = parser.parse_args()
    if args.sample_mod < 1:
        parser.error("--sample-mod must be >= 1")
        protected = [Path("data") / d / s for d in ("movielens", "netflix") for s in ("dev", "full")]
    if args.out_dir.resolve() in [p.resolve() for p in protected]:
        parser.error("Refusing to overwrite the data.py splits")

    start = time.perf_counter()
    item_counts = Counter()
    user_ids, movie_ids, ratings, timestamps = [], [], [], []
    scanned = 0
    for uid, mid, rating, ts in iter_netflix_ratings(args.netflix_dir):
        item_counts[mid] += 1  # Movie frequency in the COMPLETE Netflix dataset.
        scanned += 1
        if uid % args.sample_mod == 0:
            user_ids.append(uid)
            movie_ids.append(mid)
            ratings.append(rating)
            timestamps.append(ts)

    print(f"Scanned {scanned:,} Netflix ratings")
    print(f"Collected {len(ratings):,} ratings for sampled user IDs")

    df = pd.DataFrame({
        "userId": np.asarray(user_ids, dtype="int32"),
        "movieId": np.asarray(movie_ids, dtype="int32"),
        "rating": np.asarray(ratings, dtype="float32"),
        "timestamp": np.asarray(timestamps, dtype="int64"),
    })
    del user_ids, movie_ids, ratings, timestamps

    # Mirror data.py: remove rarely rated movies first, then eligible users.
    eligible_movies = {mid for mid, count in item_counts.items()
                       if count >= MIN_RATINGS_PER_ITEM}
    df = df[df["movieId"].isin(eligible_movies)].copy()
    grouped = df.groupby("userId")["timestamp"]
    eligible = (grouped.size() >= MIN_RATINGS_PER_USER) & (
        (grouped.max() - grouped.min()) >= MIN_SPAN_DAYS * 86400
    )
    df = df[df["userId"].isin(eligible.index[eligible])].copy()
    if df.empty:
        raise RuntimeError("No eligible users; check sampling and filters")

    # Use identical per-user chronological 80/10/remaining rules from data.py.
    # Same-day ties are ordered by movieId, because Netflix has no time-of-day.
    parts = chronological_split(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for label, part in zip(("train", "val", "test"), parts):
        path = args.out_dir / f"{label}.parquet"
        part.to_parquet(path, index=False)
        print(f"{label:5s}: {len(part):>9,} ratings | "
              f"{part.userId.nunique():>5,} users | hash {data_hash(part)} | {path}")

    train, val, test = parts
    active_days = train.assign(day=train["timestamp"] // 86400).groupby("userId")["day"].nunique()
    span = train.groupby("userId")["timestamp"].agg(lambda x: (x.max() - x.min()) / 86400)
    print("\nTRAINING HISTORY QUALITY")
    print(f"Eligible sampled users: {len(active_days):,}")
    print(f"Median training active days: {active_days.median():.1f}")
    print(f"Users with <=5 training active days: {(active_days <= 5).mean():.2%}")
    print(f"Median training history span: {span.median():.1f} days")
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    print("NOTE: Netflix has day-level timestamps. Train/val/test may share a day; "
          "do not interpret within-day ordering as real chronology.")
    print("NOTE: This is a deterministic ~1% user sample, not the full Netflix dataset.")


if __name__ == "__main__":
    main()