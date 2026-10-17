import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "compare_result"
NETFLIX_DIR = "data/netflix_prize"
NETFLIX_FILES = [f"combined_data_{i}.txt" for i in range(1, 5)]
ML_FILE = "data/movielens/ratings.csv"

NETFLIX_KEEP_MOD = 10   # keep users with userId % 10 == 0 (about 10 percent of users)
ML_KEEP_MOD = 1         # 1 keeps all MovieLens users, raise it if memory is tight
MIN_RATINGS = 20
MIN_SPAN_DAYS = 180
SEED = 42


# Loaders: both return the same schema (userId, movieId, rating, timestamp)
def load_netflix(keep_mod):
    frames, last_movie = [], None
    for name in NETFLIX_FILES:
        path = f"{NETFLIX_DIR}/{name}"
        print("reading", path)
        reader = pd.read_csv(path, header=None, names=["a", "rating", "date"],
                             dtype=str, chunksize=5_000_000)
        for chunk in reader:
            is_header = chunk["rating"].isna()
            movie = chunk["a"].where(is_header).str.rstrip(":")
            if pd.isna(movie.iloc[0]) and last_movie is not None:
                movie.iloc[0] = last_movie
            movie = movie.ffill()
            last_movie = movie.iloc[-1]
            rows = chunk[~is_header]
            user = rows["a"].astype("int64")
            keep = (user % keep_mod == 0).values
            part = pd.DataFrame({
                "userId": user[keep].values,
                "movieId": movie[~is_header][keep].astype("int32").values,
                "rating": rows["rating"][keep].astype("float32").values,
                "timestamp": pd.to_datetime(rows["date"][keep].values),
            })
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_movielens(keep_mod):
    df = pd.read_csv(ML_FILE)
    df = df[df["userId"] % keep_mod == 0].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["rating"] = df["rating"].astype("float32")
    return df[["userId", "movieId", "rating", "timestamp"]]


# Metrics
def user_table(df):
    d = df.assign(day=df["timestamp"].dt.normalize())
    g = d.groupby("userId")
    t = pd.DataFrame({
        "n": g.size(),
        "span_days": (g["timestamp"].max() - g["timestamp"].min()).dt.days,
        "active_days": g["day"].nunique(),
    })
    first_day = d["day"] == g["day"].transform("min")
    t["first_day_share"] = first_day.groupby(d["userId"]).mean()
    return t


def half_gap(d, shuffle, rng):
    """Absolute gap in mean rating between the early and late half of each user's history."""
    d = d[["userId", "rating", "timestamp"]].copy()
    if shuffle:
        d["key"] = rng.random(len(d))
    else:
        d["key"] = d["timestamp"].astype("int64")
    d = d.sort_values(["userId", "key"])
    pos = d.groupby("userId").cumcount()
    size = d.groupby("userId")["rating"].transform("size")
    d["late"] = (pos / size) >= 0.5
    m = d.groupby(["userId", "late"])["rating"].mean().unstack()
    return (m[True] - m[False]).abs().dropna()


def cold_start_share(df):
    """Share of test ratings (latest 10 percent per user) on movies never seen in training."""
    d = df.sort_values(["userId", "timestamp"])
    pos = d.groupby("userId").cumcount()
    size = d.groupby("userId")["rating"].transform("size")
    train, test = d[(pos / size) < 0.9], d[(pos / size) >= 0.9]
    seen = set(train["movieId"].unique())
    return float((~test["movieId"].isin(seen)).mean())


def summarize(df, name, rng):
    users = user_table(df)
    qualified = users[(users["n"] >= MIN_RATINGS) & (users["span_days"] >= MIN_SPAN_DAYS)]
    q_ids = set(qualified.index)
    dq = df[df["userId"].isin(q_ids)]

    real_gap = half_gap(dq, False, rng)
    null_gap = half_gap(dq, True, rng)

    n_users, n_items = df["userId"].nunique(), df["movieId"].nunique()
    return {
        "dataset": name,
        "ratings": len(df),
        "users": n_users,
        "movies": n_items,
        "mean_rating": round(float(df["rating"].mean()), 3),
        "sparsity_pct": round(100 * (1 - len(df) / (n_users * n_items)), 3),
        "first_date": df["timestamp"].min().date(),
        "last_date": df["timestamp"].max().date(),
        "median_ratings_per_user": float(users["n"].median()),
        "median_span_days": float(users["span_days"].median()),
        "median_active_days": float(users["active_days"].median()),
        "median_first_day_share": round(float(users["first_day_share"].median()), 3),
        "qualified_users": len(qualified),
        "qualified_pct": round(100 * len(qualified) / n_users, 1),
        "same_timestamp_ties_pct": round(
            100 * float(df.duplicated(["userId", "timestamp"], keep=False).mean()), 1),
        "cold_movie_share_in_test": round(cold_start_share(dq), 4),
        "half_gap_real": round(float(real_gap.mean()), 4),
        "half_gap_shuffled": round(float(null_gap.mean()), 4),
        "drift_ratio_real_over_shuffled": round(float(real_gap.mean() / null_gap.mean()), 3),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    netflix = load_netflix(NETFLIX_KEEP_MOD)
    ml = load_movielens(ML_KEEP_MOD)

    # 1. Display the raw data
    print("\nNetflix sample\n", netflix.head(), "\n")
    print("MovieLens sample\n", ml.head(), "\n")

    # 2. Side by side summary
    summary = pd.DataFrame([summarize(netflix, "Netflix", rng),
                            summarize(ml, "MovieLens", rng)]).set_index("dataset").T
    print(summary, "\n")
    summary.to_csv(f"{OUTPUT_DIR}/summary.csv")

    # 3. Rating distribution (MovieLens half stars are rounded up to whole stars)
    dist = pd.DataFrame({
        "Netflix": netflix["rating"].round().value_counts(normalize=True).sort_index(),
        "MovieLens": np.ceil(ml["rating"]).value_counts(normalize=True).sort_index(),
    })
    print(dist.round(3), "\n")
    dist.plot(kind="bar", title="Rating distribution (share of ratings)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rating_distribution.png")
    plt.close()

    # 4. User history shape
    for df, label in [(netflix, "Netflix"), (ml, "MovieLens")]:
        u = user_table(df)
        u["span_days"].plot(kind="hist", bins=60, alpha=0.5, logy=True, label=label)
    plt.legend()
    plt.title("Days between first and last rating, per user")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/user_span_days.png")
    plt.close()

    for df, label in [(netflix, "Netflix"), (ml, "MovieLens")]:
        u = user_table(df)
        u["first_day_share"].plot(kind="hist", bins=40, alpha=0.5, label=label)
    plt.legend()
    plt.title("Share of a user's ratings made on their first active day")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/first_day_share.png")
    plt.close()

    # 5. Global drift: monthly mean rating and volume
    fig, ax = plt.subplots(2, 1, sharex=False, figsize=(8, 7))
    for df, label in [(netflix, "Netflix"), (ml, "MovieLens")]:
        m = df.set_index("timestamp")["rating"].resample("MS").agg(["mean", "count"])
        m["mean"].plot(ax=ax[0], label=label)
        m["count"].plot(ax=ax[1], label=label, logy=True)
    ax[0].set_title("Mean rating by month")
    ax[1].set_title("Ratings per month (log scale)")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_mean_rating.png")
    plt.close()

    print(f"Saved summary.csv and four PNG charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()