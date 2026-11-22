import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from changepoint import (before_after_gap, changepoint_weights, fit_regime_prior, genre_shift,
                         mean_interval, regime_weights, shuffle_within_users, time_decay_weights)
from data import RATING_SCALE, SEED, load_dir, load_split, set_seed
from metrics import per_user_scores, rmse
import plots

RESULTS_DIR = Path(__file__).parent / "results"
ALL_MODELS = ["bias", "mf", "mf-refit", "mf-decay", "decay-refit", "timesvdpp", "flexrecsys-v1",
              "flexrecsys-soft", "flexrecsys-level", "flexrecsys"]
DEFAULT_MODELS = [m for m in ALL_MODELS
                  if m not in ("flexrecsys-v1", "flexrecsys-soft", "flexrecsys-level")]
# soft update strengths, as multiples of the user-refit ridge strength
SHIFT_GRID = [0.3, 1, 3, 10, 30, 100, 300]
OURS = "flexrecsys"
USER_REG_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
RECIPES = {"wd": dict(factors=32, reg=0.0, lr=3e-3, epochs=8, mf_wd=1e-4),
           "old": dict(factors=64, reg=0.05, lr=5e-3, epochs=30, mf_wd=0.0)}
LABELS = {"bias": "Bias baseline", "mf": "MF", "mf-refit": "MF, users refit",
          "mf-decay": "MF + time decay", "decay-refit": "Time decay, users refit",
          "timesvdpp": "TimeSVD++", "flexrecsys-v1": "flexrecsys v1 (window BOCPD)",
          "flexrecsys-soft": "flexrecsys, soft update",
          "flexrecsys-level": "flexrecsys, level update", "flexrecsys": "flexrecsys (ours)"}


def make_mf(args, verbose):
    import mf
    return mf.MF(args.factors, args.reg, args.lr, args.batch_size, args.epochs,
                 rating_scale=tuple(args.rating_scale), verbose=verbose,
                 weight_decay=args.mf_wd)


def outputs(m, ev, args):
    # ev.heldout is val or test, ev.history is everything the user is known to have rated before it
    return m.predict(ev.heldout), m.recommend(ev.history, ev.heldout.userId.unique(), max(args.k))


def run_bias(train, val, args, cache, ev):
    import mf
    m = mf.BiasBaseline(rating_scale=tuple(args.rating_scale)).fit(train)
    return *outputs(m, ev, args), {}


def run_mf(train, val, args, cache, ev):
    m = base_mf(train, val, args, cache)
    return *outputs(m, ev, args), {"epochs_run": len(m.history)}


def run_mf_decay(train, val, args, cache, ev):
    w = time_decay_weights(train, args.half_life, args.decay_floor)
    m = make_mf(args, args.verbose).fit(train, val, weights=w)
    return *outputs(m, ev, args), {"epochs_run": len(m.history),
                                   "mean_weight": float(w.mean())}


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


def run_mf_refit(train, val, args, cache, ev):
    # same refit as flexrecsys with all weights 1, so the refit itself isn't credited to the weights
    m = refit(base_mf(train, val, args, cache), train, None, args)
    return *outputs(m, ev, args), {"user_reg": m.user_reg_}


def run_decay_refit(train, val, args, cache, ev):
    w = time_decay_weights(train, args.half_life, args.decay_floor)
    m = refit(base_mf(train, val, args, cache), train, w, args)
    return *outputs(m, ev, args), {"mean_weight": float(w.mean())}


def detect(base, train, args):
    if args.regime_prior == "auto":
        reg, noise = fit_regime_prior(base, train, args.cp_components)
        noise = noise if args.cp_noise is None else args.cp_noise
    else:
        reg, noise = args.cp_reg, args.cp_noise
    w, diag = regime_weights(base, train, prior=args.cp_prior, n_components=args.cp_components,
                             reg=reg, min_seg=args.cp_min_seg, floor=args.regime_floor,
                             noise=noise, return_diagnostics=True)
    return w, diag, reg


def regime_fit(base, train, args, cache):
    # detector output, computed once per seed and shared by every flexrecsys variant
    if "regime" not in cache:
        cache["regime"] = detect(base, train, args)
        cache["cp_diag"] = cache["regime"][1]
    return cache["regime"]


def diagnose(train, val, data_name, args, cache):
    base = base_mf(train, val, args, cache)
    diag = regime_fit(base, train, args, cache)[1]
    flagged = set(diag.userId[diag.p_change > 0.5])
    placebo = detect(base, shuffle_within_users(train), args)[1]
    out = {"flagged_share": float((diag.p_change > 0.5).mean()),
           "flagged_share_shuffled": float((placebo.p_change > 0.5).mean())}
    print("\n[diagnostics]")
    print(f"  flagged {out['flagged_share']:.1%} of users; "
          f"{out['flagged_share_shuffled']:.1%} after shuffling each user's rating order "
          f"(all false alarms)")

    def groups(series, key):
        for name, part in (("flagged", series[series.index.isin(flagged)]),
                           ("unflagged", series[~series.index.isin(flagged)])):
            if len(part):
                m, lo, hi = out[f"{key}_{name}"] = mean_interval(part)
                print(f"    {name:9s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={len(part)}")

    print("  validation error from the ratings before the change minus after it "
          "(positive: post-change ratings predict better)")
    groups(before_after_gap(base, train, val, diag, base.user_reg_), "before_after")

    if data_name.startswith("movielens"):
        from data import load_movies
        shift, moves = genre_shift(train, load_movies("movielens"), diag)
        print("  genre mix moved at the change, beyond random splits of the same ratings")
        groups(shift.excess, "genre_excess")
        beyond = shift.beyond_chance.groupby(shift.index.isin(flagged)).mean()
        out["genre_beyond_chance"] = {"flagged": float(beyond.get(True, np.nan)),
                                      "unflagged": float(beyond.get(False, np.nan))}
        print(f"    users beyond chance (5% expected): flagged {beyond.get(True, np.nan):.1%}, "
              f"unflagged {beyond.get(False, np.nan):.1%}")
        top = moves[moves.index.isin(flagged)].abs().mean().nlargest(5)
        out["genres_moving_most"] = top.round(4).to_dict()
        print("    genres moving most for flagged users: "
              + ", ".join(f"{g} {v:.3f}" for g, v in top.items()))
    return out


def run_flexrecsys_soft(train, val, args, cache, ev):
    # a subtle shift moves a user's taste a little, it does not replace it. each user starts from
    # their full-history vector and moves toward their post-change ratings; the strength of the
    # pull back to full history is picked on val. the largest strength is close to mf-refit, so
    # landing there means the detected regimes carry nothing usable
    base = base_mf(train, val, args, cache)
    w, diag, reg = regime_fit(base, train, args, cache)
    plain = refit(base, train, None, args)
    users = val.userId.unique()
    best = None
    for mult in SHIFT_GRID:
        m = copy.deepcopy(base).refit_users(train, w, reg=base.user_reg_,
                                            shift=mult * base.user_reg_, anchor=plain)
        if args.user_reg_metric == "rmse":
            score = -rmse(val.rating, m.predict(val))
        else:
            score = per_user_scores(m.recommend(train, users, 10), val, [10])["ndcg@10"].mean()
        if args.verbose:
            print(f"    soft update x{mult:g}: {args.user_reg_metric} {abs(score):.4f}")
        if best is None or score > best[0]:
            best = (score, mult, m)
    _, mult, m = best
    edge = " (largest strength: regimes not used)" if mult == SHIFT_GRID[-1] else ""
    print(f"  soft update strength -> x{mult:g}{edge}", flush=True)
    extra = {"shift_mult": float(mult), "mean_weight": float(w.mean()),
             "users_flagged": int((diag.p_change > 0.5).sum()), "users": len(diag)}
    return *outputs(m, ev, args), extra


def run_flexrecsys_level(train, val, args, cache, ev):
    # taste vectors from the full history, exactly as mf-refit. only the rating level (user bias)
    # is re-estimated with the regime weights, so no rating is discarded for taste and the
    # ranking is identical to mf-refit by construction. nothing is tuned here
    base = base_mf(train, val, args, cache)
    w, diag, reg = regime_fit(base, train, args, cache)
    m = refit(base, train, None, args).refit_user_bias(train, w, reg=base.user_reg_)
    extra = {"mean_weight": float(w.mean()),
             "users_flagged": int((diag.p_change > 0.5).sum()), "users": len(diag)}
    return *outputs(m, ev, args), extra


def run_flexrecsys(train, val, args, cache, ev):
    # stage 1: plain MF. stage 2: locate each user's change in rating space.
    # stage 3: refit only the user vectors with the regime weights, items stay as trained
    base = base_mf(train, val, args, cache)
    w, diag, reg = regime_fit(base, train, args, cache)
    m = refit(base, train, w, args)
    extra = {"mean_weight": float(w.mean()), "detector_reg": float(reg),
             "users_flagged": int((diag.p_change > 0.5).sum()), "users": len(diag)}
    return *outputs(m, ev, args), extra


def run_flexrecsys_v1(train, val, args, cache, ev):
    base = base_mf(train, val, args, cache)
    w, diag = changepoint_weights(base, train, window=args.window, hazard=1 / args.expected_run,
                                  floor=args.cp_floor, n_components=args.cp_components,
                                  reg=args.cp_reg, return_diagnostics=True)
    cache.setdefault("cp_diag", diag)
    # stage 2: refit with change-point weights
    m = make_mf(args, args.verbose).fit(train, val, weights=w)
    extra = {"epochs_run": len(m.history), "mean_weight": float(w.mean()),
             "users_flagged": int((diag.p_change > 0.5).sum()), "users": len(diag)}
    return *outputs(m, ev, args), extra


def run_timesvdpp(train, val, args, cache, ev):
    from evaluate_timesvdpp_ranking import recommend_timesvdpp
    from timesvdpp_full import evaluate, fit_timesvdpp, setup_frames
    d = setup_frames(train, val)
    model, _, best_epoch, _ = fit_timesvdpp(d, args.tsvd_factors, args.tsvd_epochs,
                                            args.batch_size, args.tsvd_lr, args.tsvd_wd,
                                            args.device, tuple(args.rating_scale),
                                            verbose=args.verbose, ablate=args.tsvd_ablate)
    seen = None
    if ev.name == "test":
        # same training ratings, refit for the epoch count chosen on val with test timestamps
        # encoded, so test never picks the checkpoint
        d = setup_frames(train, ev.heldout)
        model, _, _, _ = fit_timesvdpp(d, args.tsvd_factors, best_epoch, args.batch_size,
                                       args.tsvd_lr, args.tsvd_wd, args.device,
                                       tuple(args.rating_scale), verbose=args.verbose,
                                       ablate=args.tsvd_ablate, select=False)
        seen = val
    _, pred = evaluate(model, d["v"], args.device, rating_scale=tuple(args.rating_scale),
                       return_pred=True)
    users = sorted(int(u) for u in ev.heldout.userId.unique())
    recs = recommend_timesvdpp(model, d, users, k=max(args.k), device=args.device,
                               rating_scale=tuple(args.rating_scale), verbose=False,
                               clamp=not args.tsvd_raw_rank, seen_extra=seen)
    return pred, recs, {"best_epoch": best_epoch}


RUNNERS = {"bias": run_bias, "mf": run_mf, "mf-refit": run_mf_refit, "mf-decay": run_mf_decay,
           "decay-refit": run_decay_refit, "timesvdpp": run_timesvdpp,
           "flexrecsys-v1": run_flexrecsys_v1, "flexrecsys-soft": run_flexrecsys_soft,
           "flexrecsys-level": run_flexrecsys_level,
           "flexrecsys": run_flexrecsys}


def load_data(args):
    if args.synthetic:
        from synthetic import make_split
        train, val, test, truth = make_split(n_users=args.syn_users, n_items=args.syn_items,
                                             shift_frac=args.shift_frac[0])
        return train, val, test, "synthetic", truth
    if args.data:
        train, val, test = load_dir(args.data)
        return train, val, test, Path(args.data).name, None
    if args.fresh_users:
        # users outside the dev sample: nothing was ever tuned or scored on them
        dev_users = load_split(args.dataset, "dev")[0].userId.unique()
        parts = load_split(args.dataset, "full")
        pool = np.setdiff1d(parts[0].userId.unique(), dev_users)
        keep = np.random.default_rng(SEED).choice(pool, size=min(args.fresh_users, len(pool)),
                                                  replace=False)
        train, val, test = (p[p.userId.isin(keep)].reset_index(drop=True) for p in parts)
        return train, val, test, f"{args.dataset}_fresh{len(keep)}", None
    train, val, test = load_split(args.dataset, args.split)
    return train, val, test, f"{args.dataset}_{args.split}", None


def detection_quality(diag, train, truth):
    # synthetic only: does p_change separate users whose planted shift lands inside train?
    from sklearn.metrics import roc_auc_score
    n_train = train.groupby("userId").size().rename("n_train")
    t = truth.set_index("userId").join(n_train)
    t["shift_in_train"] = t.shifted & (t.shift_index < t.n_train - 5)
    d = diag.set_index("userId").join(t["shift_in_train"]).dropna()
    flagged = d.p_change > 0.5
    y = d.shift_in_train.astype(bool)
    auc = float(roc_auc_score(y, d.p_change)) if y.nunique() == 2 else float("nan")
    return {"auc": auc,
            "precision": float((flagged & y).sum() / max(flagged.sum(), 1)),
            "recall": float((flagged & y).sum() / max(y.sum(), 1)),
            "users_with_shift_in_train": int(y.sum()), "users": int(len(d))}


def gain_vs_refit(per_user, model, metric="recall@10"):
    # % change over mf-refit, paired by user after averaging seeds, with a 95% interval
    w = per_user.groupby(["userId", "model"])[metric].mean().unstack()
    d = (w[model] - w["mf-refit"]).dropna()
    base = w["mf-refit"].reindex(d.index).mean()
    return tuple(100 * v / base for v in mean_interval(d))


def shift_point(out, per_user, models):
    # one point of the shift curve: excess flagged share (flagged minus shuffled) and the gains
    d = out.get("diagnostics", {})
    row = {"excess_flagged": d.get("flagged_share", np.nan) - d.get("flagged_share_shuffled", np.nan)}
    for m in models:
        if m.startswith("flexrecsys") and m in set(per_user.model):
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = gain_vs_refit(per_user, m)
    return row


def run_shift_sweep(args):
    # synthetic runs at several shares of shifted users, then the curve of gain over mf-refit,
    # with MovieLens and Netflix added where results/ml_diagnose and netflix_diagnose exist
    base_tag = args.tag or "syn_shift"
    rows = []
    for frac in args.shift_frac:
        a = copy.copy(args)
        a.shift_frac, a.tag = [frac], f"{base_tag}_{round(frac * 100)}"
        print(f"\n========== {frac:.0%} of synthetic users shifted ==========")
        out = run(a)
        row = shift_point(out, pd.read_csv(RESULTS_DIR / f"{a.tag}_per_user.csv"), args.models)
        det = out.get("changepoint_detection") or [{}]
        row.update(shift_frac=frac, auc=float(np.nanmean([x.get("auc", np.nan) for x in det])))
        rows.append(row)
    curve = pd.DataFrame(rows)

    refs = []
    for name, ref in (("MovieLens", "ml_diagnose"), ("Netflix", "netflix_diagnose")):
        js, pu = RESULTS_DIR / f"{ref}.json", RESULTS_DIR / f"{ref}_per_user.csv"
        if js.exists() and pu.exists():
            refs.append({"dataset": name, **shift_point(json.loads(js.read_text()),
                                                        pd.read_csv(pu), ["flexrecsys"])})
    refs = pd.DataFrame(refs)

    curve.to_csv(RESULTS_DIR / f"{base_tag}_curve.csv", index=False)
    paths = plots.plot_shift_curve(curve, refs, [m for m in args.models if m in curve],
                                   RESULTS_DIR / f"{base_tag}_curve")
    cols = ["shift_frac", "excess_flagged", "auc"] + [m for m in args.models if m in curve]
    print("\nRecall@10 change over mf-refit (%), by share of synthetic users shifted")
    print(curve[cols].round(3).to_string(index=False))
    if len(refs):
        print(refs[["dataset", "excess_flagged", "flexrecsys"]].round(3).to_string(index=False))
    print(f"saved {base_tag}_curve.csv, " + ", ".join(p.name for p in paths))
    return curve


def run(args):
    train, val, test, data_name, truth = load_data(args)
    if args.eval == "test":
        ev = SimpleNamespace(name="test", heldout=test,
                             history=pd.concat([train, val], ignore_index=True))
    else:
        ev = SimpleNamespace(name="val", heldout=val, history=train)
    tag = args.tag or f"compare_{data_name}" + ("_test" if args.eval == "test" else "")
    print(f"{data_name}: {len(train):,} train / {len(val):,} val / {len(test):,} test ratings, "
          f"{train.userId.nunique():,} users, {train.movieId.nunique():,} movies. "
          f"Scoring on {ev.name}.")

    runs = {m: [] for m in args.models}
    per_user, detection, flags, diagnostics = [], [], [], None
    for seed in args.seeds:
        cache = {}
        # deterministic models only need one seed
        for name in args.models:
            if name == "bias" and seed != args.seeds[0]:
                continue
            set_seed(seed)
            print(f"\n[{LABELS[name]}] seed {seed}", flush=True)
            start = time.time()
            pred, recs, extra = RUNNERS[name](train, val, args, cache, ev)
            scores = per_user_scores(recs, ev.heldout, args.k)
            # the key stays val_rmse in both modes so plots.py reads old and new files alike
            row = {"seed": seed, "val_rmse": rmse(ev.heldout.rating, pred),
                   **scores.drop(columns="userId").mean().to_dict(),
                   "n_users": len(scores), "seconds": round(time.time() - start, 1), **extra}
            runs[name].append(row)
            se = pd.DataFrame({"userId": ev.heldout.userId.to_numpy(),
                               "mse": (ev.heldout.rating.to_numpy() - np.asarray(pred)) ** 2})
            scores = scores.merge(se.groupby("userId", as_index=False).mse.mean(), on="userId")
            per_user.append(scores.assign(model=name, seed=seed))
            print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in row.items()), flush=True)
        if "cp_diag" in cache:
            flags.append(cache["cp_diag"].assign(seed=seed))
        if truth is not None and "cp_diag" in cache:
            detection.append({"seed": seed, **detection_quality(cache["cp_diag"], train, truth)})
            print("  change-point detection vs planted shifts:", detection[-1])
        if (not args.no_diagnose and seed == args.seeds[0]
                and any(m.startswith("flexrecsys") for m in args.models)):
            set_seed(seed)
            diagnostics = diagnose(train, val, data_name, args, cache)

    out = {"tag": tag, "dataset": data_name, "ours": OURS if OURS in args.models else None,
           "labels": {m: LABELS[m] for m in args.models}, "ks": args.k,
           "eval": ev.name,
           "data": {"train_ratings": len(train), "val_ratings": len(val),
                    "heldout_ratings": len(ev.heldout),
                    "users": int(train.userId.nunique()), "movies": int(train.movieId.nunique())},
           "settings": {k: v for k, v in vars(args).items() if k != "models"},
           "runs": runs}
    if detection:
        out["changepoint_detection"] = detection
    if diagnostics:
        out["diagnostics"] = diagnostics

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
    src.add_argument("--data", help="folder with train/val/test.parquet, e.g. netflix_day_dev "
                                    "(looked up under data/ if not a path)")
    src.add_argument("--synthetic", action="store_true", help="planted-shift synthetic data")
    ap.add_argument("--dataset", default="movielens", choices=list(RATING_SCALE))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--fresh-users", type=int, default=0,
                    help="use this many users from the full split that are not in dev")
    ap.add_argument("--eval", default="val", choices=["val", "test"],
                    help="split to score on; test is for the final frozen configuration only")
    ap.add_argument("--rating-scale", type=float, nargs=2, default=None)
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS,
                    help="default: every main model, or mf-refit and flexrecsys for a shift sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=[575, 576, 577])
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--tag", help="output name under results/ (default compare_<data>)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true", help="print per-epoch training lines")
    g = ap.add_argument_group("MF (shared by mf, mf-decay, flexrecsys)")
    g.add_argument("--recipe", default="wd", choices=list(RECIPES),
                   help="wd: Adam weight decay, the final configuration (default). "
                        "old: the original per-batch L2 recipe")
    g.add_argument("--factors", type=int)
    g.add_argument("--reg", type=float)
    g.add_argument("--lr", type=float)
    g.add_argument("--batch-size", type=int, default=4096)
    g.add_argument("--epochs", type=int)
    g.add_argument("--mf-wd", type=float, help="Adam weight decay for MF")
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
    g.add_argument("--no-diagnose", action="store_true",
                   help="skip the flag checks (placebo, before/after, genre shift on MovieLens) "
                        "that run on the first seed whenever a flexrecsys model is included")
    g.add_argument("--regime-prior", default="auto", choices=["auto", "fixed"],
                   help="auto fits the detector's prior strength and noise to the training ratings; "
                        "fixed uses --cp-reg and --cp-noise")
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
    g.add_argument("--shift-frac", type=float, nargs="+", default=[0.5],
                   help="share of users with a planted shift; several values (e.g. "
                        "0 0.25 0.5 0.75 1) run a sweep and plot the gain over mf-refit against it")
    g.add_argument("--syn-users", type=int, default=1500)
    g.add_argument("--syn-items", type=int, default=1200)
    args = ap.parse_args(argv)
    if args.models is None:
        sweep = args.synthetic and len(args.shift_frac) > 1
        args.models = ["mf-refit", "flexrecsys", "flexrecsys-soft"] if sweep else DEFAULT_MODELS
    for key, value in RECIPES[args.recipe].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.data and not Path(args.data).exists() and (Path("data") / args.data).exists():
        args.data = str(Path("data") / args.data)
    if args.rating_scale is None:
        netflix = args.data and "netflix" in str(args.data)
        args.rating_scale = list(RATING_SCALE["netflix" if netflix else args.dataset])
        if args.synthetic:
            args.rating_scale = [0.5, 5.0]
    args.k = sorted(args.k)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.synthetic and len(args.shift_frac) > 1:
        if "mf-refit" not in args.models:
            raise SystemExit("a shift sweep compares against mf-refit, so include it in --models")
        run_shift_sweep(args)
    else:
        run(args)