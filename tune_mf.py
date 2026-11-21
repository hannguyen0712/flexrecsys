"""
Tune mf.py on the VAL split and compare it with the bias baseline

Run from the repo root:
    # MovieLens (uses data.load_split)
    python tune_mf.py --dataset movielens --split dev

    # Netflix day split (same files TimeSVD++ was scored on)
    python tune_mf.py --dataset netflix --data-dir data/netflix_day_dev --ks 5 10 20 50
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import RATING_SCALE, load_split, set_seed
from metrics import evaluate_ranking, rmse
from mf import MF, BiasBaseline

OUT_DIR = Path(__file__).parent / "results" / "tuning"


def score(model, train, val, users, ks):
    out = {"val_rmse": rmse(val.rating, model.predict(val))}
    out.update(evaluate_ranking(model.recommend(train, users, max(ks)), val, ks))
    return out


def popularity_check(train, val, mf_recs, base_recs):
    counts = train.movieId.value_counts()

    def stats(recs):
        pop = np.array([counts.get(m, 0) for items in recs.values() for m in items])
        return f"median train ratings {np.median(pop):>8.0f}   share with <50 ratings {np.mean(pop < 50):>6.1%}"

    liked = val[val.rating >= 4.0].movieId
    rel = np.array([counts.get(m, 0) for m in liked])
    print("\npopularity of recommended movies (top-20, all val users)")
    print("  bias baseline :", stats(base_recs))
    print("  tuned MF      :", stats(mf_recs))
    print(f"  held-out liked: median train ratings {np.median(rel):>8.0f}   "
          f"share with <50 ratings {np.mean(rel < 50):>6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movielens", choices=list(RATING_SCALE),
                    help="used for the rating scale (clipping) and, without --data-dir, the split")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data-dir", default=None,
                    help="folder with train.parquet and val.parquet, e.g. data/netflix_day_dev")
    ap.add_argument("--tag", default=None, help="name for output files")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--factors", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--reg", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--lr", type=float, nargs="+", default=[2e-3, 5e-3])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--no-popularity", action="store_true")
    args = ap.parse_args()
    ks = tuple(sorted(set(args.ks)))

    set_seed()
    scale = RATING_SCALE[args.dataset]
    if args.data_dir:
        d = Path(args.data_dir)
        train, val = (pd.read_parquet(d / f"{s}.parquet") for s in ("train", "val"))  # no test
        tag = args.tag or d.name
        where = str(d)
    else:
        train, val, _ = load_split(args.dataset, args.split)   # test is never used here
        tag = args.tag or f"{args.dataset}_{args.split}"
        where = f"{args.dataset}/{args.split}"
    users = val.userId.unique()
    print(f"{where}: {len(train):,} train, {len(val):,} val ratings, "
          f"{train.userId.nunique():,} users, {train.movieId.nunique():,} train movies, "
          f"rating scale {scale}, K = {ks}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"tuning_{tag}.csv"
    hist_path = OUT_DIR / f"tuning_{tag}_history.json"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    histories = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    if rows and f"recall@{ks[0]}" not in rows[0]:
        raise SystemExit(f"{csv_path} was made with different K values; use a new --tag or delete it.")

    def save():
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        hist_path.write_text(json.dumps(histories))

    # bias baseline 
    base = BiasBaseline(rating_scale=scale).fit(train)
    if not any(r["model"] == "baseline" for r in rows):
        rows.append({"model": "baseline", **score(base, train, val, users, ks)})
        save()
        print("baseline:", {k: round(v, 4) for k, v in rows[-1].items() if k != "model"})

    # MF grid
    done = {(int(r["factors"]), float(r["reg"]), float(r["lr"]))
            for r in rows if r["model"] == "mf"}
    grid = list(itertools.product(args.factors, args.reg, args.lr))
    for n, (f, reg, lr) in enumerate(grid, 1):
        if (f, reg, lr) in done:
            continue
        set_seed()
        start = time.time()
        mf = MF(f, reg, lr, args.batch_size, args.epochs, args.patience,
                rating_scale=scale, verbose=False).fit(train, val)
        best_epoch = int(np.argmin([h["val_rmse"] for h in mf.history]) + 1)
        row = {"model": "mf", "factors": f, "reg": reg, "lr": lr,
               "best_epoch": best_epoch, "epochs_run": len(mf.history),
               **score(mf, train, val, users, ks), "seconds": round(time.time() - start, 1)}
        rows.append(row)
        histories[f"f{f}_reg{reg}_lr{lr}"] = mf.history
        save()
        k0 = ks[0]
        print(f"[{n}/{len(grid)}] factors={f} reg={reg} lr={lr}  best epoch {best_epoch}  "
              f"val_rmse {row['val_rmse']:.4f}  recall@{k0} {row[f'recall@{k0}']:.4f}  "
              f"ndcg@{k0} {row[f'ndcg@{k0}']:.4f}  ({row['seconds']}s)")

    # summary
    df = pd.DataFrame(rows)
    mf_df = df[df.model == "mf"].sort_values("val_rmse")
    k0 = ks[0]
    cols = ["factors", "reg", "lr", "best_epoch", "val_rmse", f"recall@{k0}", f"ndcg@{k0}"]
    print("\ntop 5 by val RMSE")
    print(mf_df[cols].head(5).to_string(index=False))
    print(f"\ntop 5 by Recall@{k0}")
    print(mf_df.sort_values(f"recall@{k0}", ascending=False)[cols].head(5).to_string(index=False))
    print("\nbias baseline")
    print(df[df.model == "baseline"][["val_rmse", f"recall@{k0}", f"ndcg@{k0}"]].to_string(index=False))
    print(f"\nsaved {csv_path}")

    # popularity diagnosis for the best-RMSE config
    if not args.no_popularity and len(mf_df):
        b = mf_df.iloc[0]
        set_seed()
        mf = MF(int(b.factors), float(b.reg), float(b.lr), args.batch_size, args.epochs,
                args.patience, rating_scale=scale, verbose=False).fit(train, val)
        popularity_check(train, val, mf.recommend(train, users, 20), base.recommend(train, users, 20))


if __name__ == "__main__":
    main()