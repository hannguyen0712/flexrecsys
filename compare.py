import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from changepoint import changepoint_weights, regime_weights, time_decay_weights
from data import RATING_SCALE, load_dir, load_split, set_seed
from metrics import per_user_scores, rmse
import plots

RESULTS_DIR = Path(__file__).parent / "results"
ALL_MODELS = ["bias", "mf", "mf-refit", "mf-decay", "decay-refit", "timesvdpp", "flexrecsys-v1",
              "flexrecsys"]
DEFAULT_MODELS = [m for m in ALL_MODELS if m != "flexrecsys-v1"]
OURS = "flexrecsys"
USER_REG_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000]
LABELS = {"bias": "Bias baseline", "mf": "MF", "mf-refit": "MF, users refit",
          "mf-decay": "MF + time decay", "decay-refit": "Time decay, users refit",
          "timesvdpp": "TimeSVD++", "flexrecsys-v1": "flexrecsys v1 (window BOCPD)",
          "flexrecsys": "flexrecsys (ours)"}


def make_mf(args, verbose):
    import mf
    return mf.MF(args.factors, args.reg, args.lr, args.batch_size, args.epochs,
                 rating_scale=tuple(args.rating_scale), verbose=verbose)


def run_bias(train, val, args, cache):
    import mf
    m = mf.BiasBaseline(rating_scale=tuple(args.rating_scale)).fit(train)
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), {}


def run_mf(train, val, args, cache):
    m = base_mf(train, val, args, cache)
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), \
        {"epochs_run": len(m.history)}


def run_mf_decay(train, val, args, cache):
    w = time_decay_weights(train, args.half_life, args.decay_floor)
    m = make_mf(args, args.verbose).fit(train, val, weights=w)
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), \
        {"epochs_run": len(m.history), "mean_weight": float(w.mean())}


def pick_user_reg(base, train, val, args):
    # chosen on the unweighted refit only, so the choice can't favour any weighting scheme
    users = val.userId.unique()
    best, best_score = None, -float("inf")
    for reg in USER_REG_GRID:
        m = copy.deepcopy(base).refit_users(train, None, reg=reg)
        if args.user_reg_metric == "rmse":
            score = -rmse(val.rating, m.predict(val))
        else:
            score = per_user_scores(m.recommend(train, users, 10), val, [10])["ndcg@10"].mean()
        if args.verbose:
            print(f"    user-reg {reg:g}: {args.user_reg_metric} {abs(score):.4f}")
        if score > best_score:
            best, best_score = reg, score
    edge = " (edge of grid, extend USER_REG_GRID)" if best in (USER_REG_GRID[0], USER_REG_GRID[-1]) else ""
    print(f"  user-reg auto -> {best:g}, {args.user_reg_metric} {abs(best_score):.4f}{edge}", flush=True)
    return float(best)


def base_mf(train, val, args, cache):
    # plain MF, fitted once per seed and shared by every model that builds on it
    if "mf" not in cache:
        m = make_mf(args, args.verbose).fit(train, val)
        m.user_reg_ = pick_user_reg(m, train, val, args) if args.user_reg == "auto" \
            else float(args.user_reg)
        cache["mf"] = m
    return cache["mf"]


def refit(base, train, weights, args):
    return copy.deepcopy(base).refit_users(train, weights, reg=base.user_reg_)


def run_mf_refit(train, val, args, cache):
    # same refit as flexrecsys with all weights 1, so the refit itself isn't credited to the weights
    m = refit(base_mf(train, val, args, cache), train, None, args)
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), \
        {"user_reg": m.user_reg_}


def run_decay_refit(train, val, args, cache):
    w = time_decay_weights(train, args.half_life, args.decay_floor)
    m = refit(base_mf(train, val, args, cache), train, w, args)
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), \
        {"mean_weight": float(w.mean())}


def run_flexrecsys(train, val, args, cache):
    # stage 1: plain MF. stage 2: locate each user's change in rating space.
    # stage 3: refit only the user vectors with the regime weights, items stay as trained
    base = base_mf(train, val, args, cache)
    w, diag = regime_weights(base, train, prior=args.cp_prior, n_components=args.cp_components,
                             reg=args.cp_reg, min_seg=args.cp_min_seg, floor=args.regime_floor,
                             noise=args.cp_noise, return_diagnostics=True)
    cache["cp_diag"] = diag
    m = refit(base, train, w, args)
    extra = {"mean_weight": float(w.mean()),
             "users_flagged": int((diag.p_change > 0.5).sum()), "users": len(diag)}
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), extra


def run_flexrecsys_v1(train, val, args, cache):
    base = base_mf(train, val, args, cache)
    w, diag = changepoint_weights(base, train, window=args.window, hazard=1 / args.expected_run,
                                  floor=args.cp_floor, n_components=args.cp_components,
                                  reg=args.cp_reg, return_diagnostics=True)
    cache.setdefault("cp_diag", diag)
    # stage 2: refit with change-point weights
    m = make_mf(args, args.verbose).fit(train, val, weights=w)
    extra = {"epochs_run": len(m.history), "mean_weight": float(w.mean()),
             "users_flagged": int((diag.p_change > 0.5).sum()), "users": len(diag)}
    return m.predict(val), m.recommend(train, val.userId.unique(), max(args.k)), extra


def run_timesvdpp(train, val, args, cache):
    from evaluate_timesvdpp_ranking import recommend_timesvdpp
    from timesvdpp_full import evaluate, fit_timesvdpp, setup_frames
    d = setup_frames(train, val)
    model, _, best_epoch, _ = fit_timesvdpp(d, args.tsvd_factors, args.tsvd_epochs,
                                            args.batch_size, args.tsvd_lr, args.tsvd_wd,
                                            args.device, tuple(args.rating_scale),
                                            verbose=args.verbose, ablate=args.tsvd_ablate)
    _, pred = evaluate(model, d["v"], args.device, rating_scale=tuple(args.rating_scale),
                       return_pred=True)
    users = sorted(int(u) for u in val.userId.unique())
    recs = recommend_timesvdpp(model, d, users, k=max(args.k), device=args.device,
                               rating_scale=tuple(args.rating_scale), verbose=False,
                               clamp=not args.tsvd_raw_rank)
    return pred, recs, {"best_epoch": best_epoch}


RUNNERS = {"bias": run_bias, "mf": run_mf, "mf-refit": run_mf_refit, "mf-decay": run_mf_decay,
           "decay-refit": run_decay_refit, "timesvdpp": run_timesvdpp,
           "flexrecsys-v1": run_flexrecsys_v1, "flexrecsys": run_flexrecsys}


def load_data(args):
    if args.synthetic:
        from synthetic import make_split
        train, val, _, truth = make_split(n_users=args.syn_users, n_items=args.syn_items)
        return train, val, "synthetic", truth
    if args.data:
        train, val, _ = load_dir(args.data)
        return train, val, Path(args.data).name, None
    train, val, _ = load_split(args.dataset, args.split)
    return train, val, f"{args.dataset}_{args.split}", None


def detection_quality(diag, train, truth):
    # synthetic only: does p_change separate users whose planted shift lands inside train?
    from sklearn.metrics import roc_auc_score
    n_train = train.groupby("userId").size().rename("n_train")
    t = truth.set_index("userId").join(n_train)
    t["shift_in_train"] = t.shifted & (t.shift_index < t.n_train - 5)
    d = diag.set_index("userId").join(t["shift_in_train"]).dropna()
    flagged = d.p_change > 0.5
    y = d.shift_in_train.astype(bool)
    return {"auc": float(roc_auc_score(y, d.p_change)),
            "precision": float((flagged & y).sum() / max(flagged.sum(), 1)),
            "recall": float((flagged & y).sum() / max(y.sum(), 1)),
            "users_with_shift_in_train": int(y.sum()), "users": int(len(d))}


def run(args):
    train, val, data_name, truth = load_data(args)
    tag = args.tag or f"compare_{data_name}"
    print(f"{data_name}: {len(train):,} train / {len(val):,} val ratings, "
          f"{train.userId.nunique():,} users, {train.movieId.nunique():,} movies")

    runs = {m: [] for m in args.models}
    per_user, detection, flags = [], [], []
    for seed in args.seeds:
        cache = {}
        # deterministic models only need one seed
        for name in args.models:
            if name == "bias" and seed != args.seeds[0]:
                continue
            set_seed(seed)
            print(f"\n[{LABELS[name]}] seed {seed}", flush=True)
            start = time.time()
            pred, recs, extra = RUNNERS[name](train, val, args, cache)
            scores = per_user_scores(recs, val, args.k)
            row = {"seed": seed, "val_rmse": rmse(val.rating, pred),
                   **scores.drop(columns="userId").mean().to_dict(),
                   "n_users": len(scores), "seconds": round(time.time() - start, 1), **extra}
            runs[name].append(row)
            se = pd.DataFrame({"userId": val.userId.to_numpy(),
                               "mse": (val.rating.to_numpy() - np.asarray(pred)) ** 2})
            scores = scores.merge(se.groupby("userId", as_index=False).mse.mean(), on="userId")
            per_user.append(scores.assign(model=name, seed=seed))
            print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in row.items()), flush=True)
        if "cp_diag" in cache:
            flags.append(cache["cp_diag"].assign(seed=seed))
        if truth is not None and "cp_diag" in cache:
            detection.append({"seed": seed, **detection_quality(cache["cp_diag"], train, truth)})
            print("  change-point detection vs planted shifts:", detection[-1])

    out = {"tag": tag, "dataset": data_name, "ours": OURS if OURS in args.models else None,
           "labels": {m: LABELS[m] for m in args.models}, "ks": args.k,
           "data": {"train_ratings": len(train), "val_ratings": len(val),
                    "users": int(train.userId.nunique()), "movies": int(train.movieId.nunique())},
           "settings": {k: v for k, v in vars(args).items() if k != "models"},
           "runs": runs}
    if detection:
        out["changepoint_detection"] = detection

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"{tag}.json"
    json_path.write_text(json.dumps(out, indent=2, default=float))
    pd.concat(per_user, ignore_index=True).to_csv(RESULTS_DIR / f"{tag}_per_user.csv", index=False)
    if flags:
        pd.concat(flags, ignore_index=True).to_csv(RESULTS_DIR / f"{tag}_flags.csv", index=False)
    paths = plots.render_all(json_path)
    print(f"\nsaved {json_path.name}, {tag}_per_user.csv, " + ", ".join(p.name for p in paths))
    print("\n" + plots.comparison_table(*plots.load_results(json_path), fmt="markdown"))
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data", help="folder with train/val/test.parquet, e.g. data/netflix_day_dev")
    src.add_argument("--synthetic", action="store_true", help="planted-shift synthetic data")
    ap.add_argument("--dataset", default="movielens", choices=list(RATING_SCALE))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--rating-scale", type=float, nargs=2, default=None)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[575])
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--tag", help="output name under results/ (default compare_<data>)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true", help="print per-epoch training lines")
    g = ap.add_argument_group("MF (shared by mf, mf-decay, flexrecsys)")
    g.add_argument("--factors", type=int, default=64)
    g.add_argument("--reg", type=float, default=0.05)
    g.add_argument("--lr", type=float, default=5e-3)
    g.add_argument("--batch-size", type=int, default=4096)
    g.add_argument("--epochs", type=int, default=30)
    g = ap.add_argument_group("time decay baseline")
    g.add_argument("--half-life", type=float, default=180.0, help="days")
    g.add_argument("--decay-floor", type=float, default=0.0)
    g = ap.add_argument_group("flexrecsys v1 window BOCPD (cp-components and cp-reg are shared)")
    g.add_argument("--window", type=int, default=10, help="ratings per embedding window")
    g.add_argument("--expected-run", type=float, default=20.0, help="1/hazard, in windows")
    g.add_argument("--cp-floor", type=float, default=0.2, help="weight for pre-change ratings")
    g.add_argument("--cp-components", type=int, default=8)
    g.add_argument("--cp-reg", type=float, default=5.0)
    g = ap.add_argument_group("flexrecsys regime weighting")
    g.add_argument("--user-reg", default="auto",
                   help="ridge strength for the user refit (mf-refit, decay-refit, flexrecsys); "
                        "auto picks it per seed from USER_REG_GRID by validation NDCG@10 of mf-refit")
    g.add_argument("--user-reg-metric", default="ndcg", choices=["ndcg", "rmse"],
                   help="validation metric that auto user-reg optimises")
    g.add_argument("--cp-prior", type=float, default=0.2, help="prior P(user has a change)")
    g.add_argument("--cp-min-seg", type=int, default=10, help="fewest ratings in a regime")
    g.add_argument("--regime-floor", type=float, default=0.0, help="weight for pre-change ratings")
    g.add_argument("--cp-noise", type=float, default=None,
                   help="rating noise sd for the detector (default: base MF train residual)")
    g = ap.add_argument_group("TimeSVD++")
    g.add_argument("--tsvd-factors", type=int, default=32)
    g.add_argument("--tsvd-epochs", type=int, default=8)
    g.add_argument("--tsvd-lr", type=float, default=3e-3)
    g.add_argument("--tsvd-wd", type=float, default=1e-4)
    g.add_argument("--tsvd-ablate", nargs="*", default=[],
                   choices=["item_bin", "user_drift", "user_day", "implicit"],
                   help="switch off TimeSVD++ components")
    g.add_argument("--tsvd-raw-rank", action="store_true",
                   help="rank on raw scores instead of scores clamped to the rating scale")
    g = ap.add_argument_group("synthetic data")
    g.add_argument("--syn-users", type=int, default=1500)
    g.add_argument("--syn-items", type=int, default=1200)
    args = ap.parse_args(argv)
    if args.rating_scale is None:
        netflix = args.data and "netflix" in str(args.data)
        args.rating_scale = list(RATING_SCALE["netflix" if netflix else args.dataset])
        if args.synthetic:
            args.rating_scale = [0.5, 5.0]
    args.k = sorted(args.k)
    return args


if __name__ == "__main__":
    run(parse_args())