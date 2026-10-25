from pathlib import Path
import numpy as np
import pandas as pd

from data import data_hash

SOURCE = Path("data/netflix_dev")
OUTPUT = Path("data/netflix_day_dev")

# Reconstruct the eligible Netflix sample.
parts = [
    pd.read_parquet(SOURCE / f"{name}.parquet")
    for name in ["train", "val", "test"]
]

df = pd.concat(parts, ignore_index=True)

# Netflix timestamps encode calendar dates.
df = df.sort_values(
    ["userId", "timestamp", "movieId"]
).reset_index(drop=True)

splits = {"train": [], "val": [], "test": []}
skipped_users = 0

for user_id, group in df.groupby("userId", sort=True):

    # Each row is one distinct rating day.
    day_counts = (
        group.groupby("timestamp", sort=True)
        .size()
    )

    days = day_counts.index.to_numpy()
    counts = day_counts.to_numpy()
    n_days = len(days)

    # Three distinct days are needed to give each split
    # at least one complete rating day.
    if n_days < 3:
        skipped_users += 1
        continue

    cumulative = np.cumsum(counts)
    total = len(group)

    # Choose day boundaries near 80% and 90%
    # of the user's rating count.
    train_end = int(np.searchsorted(cumulative, 0.80 * total))
    train_end = max(0, min(train_end, n_days - 3))

    val_end = int(np.searchsorted(cumulative, 0.90 * total))
    val_end = max(train_end + 1, min(val_end, n_days - 2))

    train_days = days[:train_end + 1]
    val_days = days[train_end + 1:val_end + 1]
    test_days = days[val_end + 1:]

    splits["train"].append(
        group[group["timestamp"].isin(train_days)]
    )
    splits["val"].append(
        group[group["timestamp"].isin(val_days)]
    )
    splits["test"].append(
        group[group["timestamp"].isin(test_days)]
    )

OUTPUT.mkdir(parents=True, exist_ok=True)

result = {}

for name, frames in splits.items():
    part = pd.concat(frames, ignore_index=True)
    part = part.sort_values(
        ["userId", "timestamp", "movieId"]
    ).reset_index(drop=True)

    part.to_parquet(OUTPUT / f"{name}.parquet", index=False)
    result[name] = part

    print(
        f"{name}: {len(part):,} ratings | "
        f"{part.userId.nunique():,} users | "
        f"hash {data_hash(part)}"
    )

print("\nUsers excluded (<3 distinct days):", skipped_users)

train = result["train"]
val = result["val"]
test = result["test"]

def check_boundary(earlier, later):
    last = earlier.groupby("userId")["timestamp"].max()
    first = later.groupby("userId")["timestamp"].min()
    common = last.index.intersection(first.index)

    return int((last.loc[common] >= first.loc[common]).sum())

print("Train/val boundary violations:",
      check_boundary(train, val))

print("Val/test boundary violations:",
      check_boundary(val, test))

unseen = ~val["movieId"].isin(train["movieId"])
print("Validation ratings with unseen movies:",
      int(unseen.sum()))