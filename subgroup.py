import sys

import numpy as np
import pandas as pd

OURS = "flexrecsys"


def paired(d, n_boot=4000, seed=0):
    d = d.dropna().to_numpy()
    if len(d) == 0:
        return "no users"
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return f"{d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  n={len(d)}"


if __name__ == "__main__":
    # OURS can be overridden by the third argument
    if len(sys.argv) < 2:
        raise SystemExit("usage: python subgroup.py results/<tag> [baseline]")
    stem = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "mf-refit"
    OURS = sys.argv[3] if len(sys.argv) > 3 else OURS
    pu = pd.read_csv(f"{stem}_per_user.csv")
    p_change = pd.read_csv(f"{stem}_flags.csv").groupby("userId").p_change.mean()
    flagged = p_change.index[p_change > 0.5]
    n_train = pd.read_csv(f"{stem}_flags.csv").groupby("userId").n_ratings.first()
    q1, q2 = n_train.quantile([1 / 3, 2 / 3])
    thirds = [("short history", n_train.min(), q1), ("medium history", q1 + 1, q2),
              ("long history", q2 + 1, n_train.max())]
    print(f"{len(flagged)} of {len(p_change)} users flagged (mean p_change over seeds > 0.5)")

    metrics = [m for m in ["mse", "recall@10", "ndcg@10"] if m in pu]
    # average each user over seeds, then pair the two models on the same users
    wide = pu[pu.model.isin([OURS, base])].groupby(["userId", "model"])[metrics].mean().unstack()
    for m in metrics:
        d = wide[m][OURS] - wide[m][base]
        hit = d.index.isin(flagged)
        better = "negative" if m == "mse" else "positive"
        print(f"\n{m}: {OURS} minus {base}, 95% bootstrap interval ({better} is better)")
        print("  all users      ", paired(d))
        print("  flagged users  ", paired(d[hit]))
        print("  unflagged users", paired(d[~hit]))
        # history length: thirds of users by number of training ratings
        for label, lo, hi in thirds:
            rows = d.index.isin(n_train.index[(n_train >= lo) & (n_train <= hi)])
            print(f"  {label} ({lo:.0f}-{hi:.0f} ratings)".ljust(34), paired(d[rows]))
            print("      of which flagged".ljust(34), paired(d[rows & hit]))