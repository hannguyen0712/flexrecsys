# synthetic ratings with planted abrupt preference shifts, for sanity-checking the comparison
# python synthetic.py --users 1500 --items 1200   (writes data/synthetic/dev/*.parquet)

import argparse

import numpy as np
import pandas as pd

from data import DATA_DIR, SEED, chronological_split, data_hash

DAY = 86_400


def make_ratings(n_users=1500, n_items=1200, n_factors=8, n_genres=10, shift_frac=0.5,
                 min_ratings=60, max_ratings=250, seed=SEED):
    # items cluster around genre centres; each user likes one genre, and shifted users
    # jump to a different genre partway through their history (50-85% of the way in,
    # so the new taste is mostly visible late in train and throughout val/test)
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (n_genres, n_factors))
    genre = rng.integers(0, n_genres, n_items)
    Q = centres[genre] + rng.normal(0, 0.5, (n_items, n_factors))
    b_i = rng.normal(0, 0.3, n_items)
    start = pd.Timestamp("2015-01-01").value // 10**9

    rows, truth = [], []
    for u in range(n_users):
        n = int(rng.integers(min_ratings, max_ratings + 1))
        g1 = rng.integers(n_genres)
        g2 = (g1 + rng.integers(1, n_genres)) % n_genres
        shifted = rng.random() < shift_frac
        cut = int(n * rng.uniform(0.5, 0.85)) if shifted else n
        p1 = 0.6 * centres[g1] + rng.normal(0, 0.3, n_factors)
        p2 = 0.6 * centres[g2] + rng.normal(0, 0.3, n_factors)
        b_u = rng.normal(0, 0.3)

        span = int(rng.integers(300, 1100))
        days = np.sort(rng.integers(0, span, n))
        seen = np.zeros(n_items, bool)
        aff1, aff2 = Q @ p1, Q @ p2
        for j in range(n):
            aff = aff1 if j < cut else aff2
            # users mostly pick what they like, with some exploration
            logits = np.where(seen, -np.inf, 0.8 * aff)
            prob = np.exp(logits - logits.max())
            i = int(rng.choice(n_items, p=prob / prob.sum()))
            seen[i] = True
            r = 3.4 + b_u + b_i[i] + 0.35 * aff[i] + rng.normal(0, 0.6)
            rows.append((u + 1, i + 1, float(np.clip(np.round(r * 2) / 2, 0.5, 5.0)),
                         start + int(days[j]) * DAY))
        truth.append((u + 1, shifted, cut if shifted else -1, n))

    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
    truth = pd.DataFrame(truth, columns=["userId", "shifted", "shift_index", "n_ratings"])
    return df, truth


def make_split(**kw):
    df, truth = make_ratings(**kw)
    train, val, test = chronological_split(df)
    return train, val, test, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--items", type=int, default=1200)
    ap.add_argument("--shift-frac", type=float, default=0.5)
    args = ap.parse_args()
    train, val, test, truth = make_split(n_users=args.users, n_items=args.items,
                                         shift_frac=args.shift_frac)
    out = DATA_DIR / "synthetic" / "dev"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in zip(["train", "val", "test"], [train, val, test]):
        part.to_parquet(out / f"{name}.parquet", index=False)
        print(f"  {name:5s} {len(part):>9,} ratings  hash {data_hash(part)}")
    truth.to_csv(out / "truth.csv", index=False)
    print(f"saved {out}")
