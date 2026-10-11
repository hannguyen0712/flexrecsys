import hashlib
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SEED = 575
MIN_RATINGS_PER_USER = 50   # enough history to compare recent vs. past behavior
MIN_SPAN_DAYS = 180         # history must cover at least ~6 months
MIN_RATINGS_PER_ITEM = 5    # drop near-empty movies
DEV_USER_FRACTION = 0.10    # dev split = 10% of eligible users, for fast iteration
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10   # per-user chronological; the rest is test

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "ml-25m"
URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def download():
    if (RAW_DIR / "ratings.csv").exists():
        return
    print("Downloading MovieLens 25M (~250 MB)...")
    r = requests.get(URL, timeout=600)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(DATA_DIR)


def load_movies():
    # run movies.csv and get genres as a list, e.g. ['Comedy', 'Romance']
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    movies["genres"] = movies["genres"].str.split("|")
    return movies


def filter_ratings(df):
    counts = df["movieId"].value_counts()
    df = df[df["movieId"].isin(counts.index[counts >= MIN_RATINGS_PER_ITEM])]
    g = df.groupby("userId")["timestamp"]
    ok = (g.size() >= MIN_RATINGS_PER_USER) & (g.max() - g.min() >= MIN_SPAN_DAYS * 86_400)
    return df[df["userId"].isin(ok.index[ok])]


def chronological_split(df):
    # user's earliest ratings -> train, next -> val, latest -> test.
    df = df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    pos = df.groupby("userId").cumcount().to_numpy()
    n = df.groupby("userId")["userId"].transform("size").to_numpy()
    n_val = np.maximum(1, (n * VAL_FRAC).astype(int))
    n_train = np.minimum((n * TRAIN_FRAC).astype(int), n - n_val - 1)
    part = np.where(pos < n_train, 0, np.where(pos < n_train + n_val, 1, 2))
    return [df[part == i].reset_index(drop=True) for i in range(3)]

# fingerprints
def data_hash(df):
    raw = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:10]


def build_splits():
    ratings = filter_ratings(pd.read_csv(RAW_DIR / "ratings.csv", dtype=DTYPES))
    users = np.sort(ratings["userId"].unique())
    dev_users = np.random.default_rng(SEED).choice(
        users, int(len(users) * DEV_USER_FRACTION), replace=False
    )
    for name, df in [("dev", ratings[ratings["userId"].isin(dev_users)]), ("full", ratings)]:
        out = DATA_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{name}]")
        for split, part in zip(["train", "val", "test"], chronological_split(df)):
            part.to_parquet(out / f"{split}.parquet", index=False)
            print(f"  {split:5s} {len(part):>11,} ratings  "
                  f"{part.userId.nunique():>7,} users  hash {data_hash(part)}")


def load_split(name="dev"):
    # (train, val, test) DataFrames with columns userId, movieId, rating, timestamp
    d = DATA_DIR / name
    if not (d / "train.parquet").exists():
        raise FileNotFoundError("Splits not built yet. Run: python data.py")
    return tuple(pd.read_parquet(d / f"{s}.parquet") for s in ["train", "val", "test"])


if __name__ == "__main__":
    download()
    build_splits()
    print("\nPost the hashes above in the group chat; they should match everyone's.")
