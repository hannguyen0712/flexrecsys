# python data.py movielens
# python data.py netflix --dev-only

import argparse
import hashlib
import io
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SEED = 575
MIN_RATINGS_PER_USER = 50
MIN_SPAN_DAYS = 180
MIN_RATINGS_PER_ITEM = 5
DEV_USER_FRACTION = 0.10
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10  

DATA_DIR = Path(__file__).parent / "data"
DATASETS = ("movielens", "netflix")
RATING_SCALE = {"movielens": (0.5, 5.0), "netflix": (1.0, 5.0)}  # for clipping predictions

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
NETFLIX_KAGGLE = "netflix-inc/netflix-prize-data"
DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def raw_dir(dataset):
    return DATA_DIR / dataset / "raw"


def download_movielens():
    if (raw_dir("movielens") / "ratings.csv").exists():
        return
    print("Downloading MovieLens 25M (~250 MB)...")
    r = requests.get(MOVIELENS_URL, timeout=600)
    r.raise_for_status()
    out = raw_dir("movielens")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():  # flatten ml-25m/x.csv into raw/x.csv
            if name.endswith(".csv"):
                (out / Path(name).name).write_bytes(z.read(name))


def read_movielens():
    return pd.read_csv(raw_dir("movielens") / "ratings.csv", dtype=DTYPES)


def download_netflix():
    out = raw_dir("netflix")
    if (out / "combined_data_1.txt").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    print("Downloading Netflix Prize data from Kaggle (~700 MB zipped, ~2 GB unpacked)...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", NETFLIX_KAGGLE, "-p", str(out), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        raise SystemExit(
            "Kaggle CLI not found. Run `pip install kaggle`, set up your API token (see README),\n"
            f"or download https://www.kaggle.com/datasets/{NETFLIX_KAGGLE} by hand and unzip\n"
            f"it into {out}"
        ) from None


def parse_netflix_file(path):
    # format: a "movieId:" line, then "userId,rating,yyyy-mm-dd" lines for that movie
    df = pd.read_csv(path, header=None, names=["userId", "rating", "date"], dtype=str)
    header = df["rating"].isna()
    movie = df["userId"].where(header).str.rstrip(":").ffill()
    df = df[~header]
    return pd.DataFrame({
        "userId": df["userId"].astype("int32").to_numpy(),
        "movieId": movie[~header].astype("int32").to_numpy(),
        "rating": df["rating"].astype("float32").to_numpy(),
        # netflix only has the day, stored as midnight utc in unix seconds
        "timestamp": ((pd.to_datetime(df["date"]) - pd.Timestamp("1970-01-01"))
                      // pd.Timedelta(seconds=1)).astype("int64").to_numpy(),
    })


def read_netflix():
    files = sorted(raw_dir("netflix").glob("combined_data_*.txt"))
    if not files:
        raise FileNotFoundError(f"No combined_data_*.txt in {raw_dir('netflix')}")
    parts = []
    for f in files:
        print(f"  parsing {f.name}")
        parts.append(parse_netflix_file(f))
    return pd.concat(parts, ignore_index=True)


def load_movies(dataset="movielens"):
    # movielens: movieId, title, genres (list). netflix: movieId, year, title
    if dataset == "movielens":
        movies = pd.read_csv(raw_dir("movielens") / "movies.csv")
        movies["genres"] = movies["genres"].str.split("|")
        return movies
    rows = []
    with open(raw_dir("netflix") / "movie_titles.csv", encoding="latin-1") as f:
        for line in f:
            # titles can contain commas, so only split on the first two
            movie_id, year, title = line.rstrip("\n").split(",", 2)
            rows.append((int(movie_id), None if year == "NULL" else int(year), title))
    return pd.DataFrame(rows, columns=["movieId", "year", "title"])


def filter_ratings(df):
    counts = df["movieId"].value_counts()
    df = df[df["movieId"].isin(counts.index[counts >= MIN_RATINGS_PER_ITEM])]
    g = df.groupby("userId")["timestamp"]
    ok = (g.size() >= MIN_RATINGS_PER_USER) & (g.max() - g.min() >= MIN_SPAN_DAYS * 86_400)
    return df[df["userId"].isin(ok.index[ok])]


def chronological_split(df):
    # earliest ratings go to train, then val, latest to test
    # same-timestamp ties broken by movieId, every user gets at least one val and test rating
    df = df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    pos = df.groupby("userId").cumcount().to_numpy()
    n = df.groupby("userId")["userId"].transform("size").to_numpy()
    n_val = np.maximum(1, (n * VAL_FRAC).astype(int))
    n_train = np.minimum((n * TRAIN_FRAC).astype(int), n - n_val - 1)
    part = np.where(pos < n_train, 0, np.where(pos < n_train + n_val, 1, 2))
    return [df[part == i].reset_index(drop=True) for i in range(3)]


def data_hash(df):
    # short fingerprint so everyone can confirm they have identical splits
    raw = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:10]


def build_splits(dataset, ratings, include_full=True):
    print(f"  raw: {len(ratings):,} ratings, {ratings.userId.nunique():,} users")
    ratings = filter_ratings(ratings)
    print(f"  after filtering: {len(ratings):,} ratings, {ratings.userId.nunique():,} users")
    users = np.sort(ratings["userId"].unique())
    dev_users = np.random.default_rng(SEED).choice(
        users, int(len(users) * DEV_USER_FRACTION), replace=False
    )
    targets = [("dev", ratings[ratings["userId"].isin(dev_users)])]
    if include_full:
        targets.append(("full", ratings))
    for name, df in targets:
        out = DATA_DIR / dataset / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{dataset}/{name}]")
        for split, part in zip(["train", "val", "test"], chronological_split(df)):
            part.to_parquet(out / f"{split}.parquet", index=False)
            print(f"  {split:5s} {len(part):>11,} ratings  "
                  f"{part.userId.nunique():>7,} users  hash {data_hash(part)}")


def load_split(dataset="movielens", name="dev"):
    # returns (train, val, test) with columns userId, movieId, rating, timestamp
    d = DATA_DIR / dataset / name
    if not (d / "train.parquet").exists():
        raise FileNotFoundError(f"Splits not built yet. Run: python data.py {dataset}")
    return load_dir(d)


def load_dir(folder):
    # any folder holding train/val/test.parquet, e.g. data/netflix_day_dev
    d = Path(folder)
    if not (d / "train.parquet").exists():
        raise FileNotFoundError(f"No train.parquet in {d}")
    return tuple(pd.read_parquet(d / f"{s}.parquet") for s in ["train", "val", "test"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--dev-only", action="store_true",
                    help="skip the full split (saves memory and disk, useful for Netflix)")
    args = ap.parse_args()

    if args.dataset == "movielens":
        download_movielens()
        ratings = read_movielens()
    else:
        download_netflix()
        ratings = read_netflix()
    build_splits(args.dataset, ratings, include_full=not args.dev_only)