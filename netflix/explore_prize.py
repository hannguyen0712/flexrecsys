import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data/"
FILES = ["combined_data_1.txt"]  
MAX_ROWS = 5_000_000             # set to None to read whole files
MIN_RATINGS = 20


def load_file(path, nrows=None):
    raw = pd.read_csv(path, header=None, names=["a", "rating", "date"],
                      dtype=str, nrows=nrows)
    is_header = raw["rating"].isna()
    movie = raw["a"].where(is_header).str.rstrip(":").ffill().astype("int32")
    df = raw[~is_header].copy()
    df["movieId"] = movie[~is_header].values
    df["userId"] = df["a"].astype("int32")
    df["rating"] = df["rating"].astype("int8")
    df["timestamp"] = pd.to_datetime(df["date"])
    return df[["userId", "movieId", "rating", "timestamp"]]


df = pd.concat([load_file(f"{DATA}/{f}", MAX_ROWS) for f in FILES],
               ignore_index=True)

# Movie titles
titles = {}
with open(f"{DATA}/movie_titles.csv", encoding="latin1") as f:
    for line in f:
        mid, year, title = line.rstrip("\n").split(",", 2)
        titles[int(mid)] = (year, title)

# Basic facts
n_users, n_movies = df["userId"].nunique(), df["movieId"].nunique()
print(df.head(), "\n")
print("ratings:", len(df), "users:", n_users, "movies:", n_movies)
print("mean rating:", round(df["rating"].mean(), 3))
print("date range:", df["timestamp"].min().date(), "to", df["timestamp"].max().date())
print("sparsity %:", round(100 * (1 - len(df) / (n_users * n_movies)), 3), "\n")
print(df["rating"].value_counts(normalize=True).sort_index().round(3), "\n")

# 2. Which users could support a recent versus past comparison?
g = df.groupby("userId")["timestamp"].agg(["count", "min", "max"])
g["span_days"] = (g["max"] - g["min"]).dt.days if False else (g["max"].sub(g["min"])).dt.days
enough = g[(g["count"] >= MIN_RATINGS) & (g["span_days"] >= 180)]
print("users with", MIN_RATINGS, "or more ratings and 180 or more days of history:",
      len(enough), "of", len(g), "\n")

# 3. Drift hook: mean rating and volume by month
monthly = df.set_index("timestamp")["rating"].resample("MS").agg(["mean", "count"])
fig, ax = plt.subplots(2, 1, sharex=True)
monthly["mean"].plot(ax=ax[0], title="Mean rating by month")
monthly["count"].plot(ax=ax[1], title="Ratings per month")
plt.tight_layout()
plt.savefig("prize_monthly.png")
plt.close()

# 4. Prize style temporal split: each user's latest 10 percent is the test set
df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)
pos = df.groupby("userId").cumcount()
size = df.groupby("userId")["rating"].transform("size")
train = df[pos / size < 0.9].copy()
test = df[pos / size >= 0.9].copy()


def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred.sub(truth)) ** 2)))


mu = train["rating"].mean()
print("RMSE, global mean only:", round(rmse(pd.Series(mu, index=test.index), test["rating"]), 4))

# Baseline from the paper: mu + item bias + user bias (shrunk)
train["res"] = train["rating"].sub(mu)
si = train.groupby("movieId")["res"].agg(["sum", "count"])
b_i = si["sum"] / (si["count"] + 25)
train["res2"] = train["res"].sub(train["movieId"].map(b_i))
su = train.groupby("userId")["res2"].agg(["sum", "count"])
b_u = su["sum"] / (su["count"] + 10)

pred = (mu + test["movieId"].map(b_i).fillna(0)
        + test["userId"].map(b_u).fillna(0)).clip(1, 5)
print("RMSE, global mean plus item and user bias:", round(rmse(pred, test["rating"]), 4))

# 5. Most rated movies, as a sanity check on the titles file
top = df["movieId"].value_counts().head(10)
for mid, n in top.items():
    print(n, titles.get(mid, ("?", "unknown")))