import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bocpd import compute_residuals, compute_user_blending_weights
from mf import MF
from data import load_split, set_seed, RATING_SCALE
from metrics import rmse

# Blend long-term and short-term embeddings using BOCPD run length posterior
def run_blending_experiment(dataset, split, results_dir, n_factors=64, hazard=1/50,
                            min_short_ratings=5, ridge_reg=1.0):
    
    set_seed()
    scale = RATING_SCALE[dataset]
    train, val, _ = load_split(dataset, split)

    # train long-term MF on all data
    mf_long = MF(n_factors=n_factors, rating_scale=scale, verbose=False).fit(train, val)
    baseline_rmse = rmse(val.rating, mf_long.predict(val))

    # get blending weight and changepoint index per user
    residuals = compute_residuals(mf_long, train)
    alphas, cp_indices = compute_user_blending_weights(residuals, hazard=hazard)
    users_with_cp = sum(1 for idx in cp_indices.values() if idx > 0)

    # extract short-term ratings 
    short_term_dfs = []
    for user, cp_idx in cp_indices.items():
        user_ratings = train[train.userId == user].sort_values('timestamp')
        if cp_idx > 0 and len(user_ratings) - cp_idx >= min_short_ratings:
            short_term_dfs.append(user_ratings.iloc[cp_idx:])
        elif len(user_ratings) >= min_short_ratings:
            n = max(min_short_ratings, len(user_ratings) // 5)
            short_term_dfs.append(user_ratings.tail(n))

    # compute short-term embeddings via ridge regression with fixed Q
    short_term_df = pd.concat(short_term_dfs, ignore_index=True) if short_term_dfs else train.head(0)
    short_term_vectors = mf_long.user_vectors_from(short_term_df, reg=ridge_reg)

    # predict: p_final = α * p_long + (1-α) * p_short
    blended_preds = mf_long.predict_blended(val, short_term_vectors, alphas)
    blended_rmse = rmse(val.rating, blended_preds)

    alphas_arr = np.array(list(alphas.values()))
    results = {
        "dataset": dataset,
        "split": split,
        "method": "embedding_blending",
        "n_factors": n_factors,
        "hazard": hazard,
        "min_short_ratings": min_short_ratings,
        "ridge_reg": ridge_reg,
        "baseline_rmse": float(baseline_rmse),
        "blended_rmse": float(blended_rmse),
        "improvement_pct": float((baseline_rmse - blended_rmse) / baseline_rmse * 100),
        "n_users": len(alphas),
        "users_with_changepoints": users_with_cp,
        "users_with_short_embed": len(short_term_vectors),
        "alpha_mean": float(alphas_arr.mean()),
        "alpha_std": float(alphas_arr.std()),
        "alpha_lt_0.3": int((alphas_arr < 0.3).sum()),
        "alpha_gt_0.7": int((alphas_arr > 0.7).sum()),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"bocpd_blending_{dataset}_{split}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

# Tuning the hazard parameter to find the best configuration for embedding blending.
def run_sweep(dataset, split, results_dir):
    set_seed()
    scale = RATING_SCALE[dataset]
    train, val, _ = load_split(dataset, split)

    mf_long = MF(n_factors=64, rating_scale=scale, verbose=False).fit(train, val)
    baseline_rmse = rmse(val.rating, mf_long.predict(val))
    residuals = compute_residuals(mf_long, train)

    sweep_results = []
    for hazard in [1/20, 1/30, 1/50, 1/75, 1/100]:
        alphas, cp_indices = compute_user_blending_weights(residuals, hazard=hazard)

        short_term_dfs = []
        for user, cp_idx in cp_indices.items():
            user_ratings = train[train.userId == user].sort_values('timestamp')
            if cp_idx > 0 and len(user_ratings) - cp_idx >= 5:
                short_term_dfs.append(user_ratings.iloc[cp_idx:])
            elif len(user_ratings) >= 5:
                short_term_dfs.append(user_ratings.tail(max(5, len(user_ratings) // 5)))

        short_term_df = pd.concat(short_term_dfs, ignore_index=True)
        short_term_vectors = mf_long.user_vectors_from(short_term_df, reg=1.0)

        blended_preds = mf_long.predict_blended(val, short_term_vectors, alphas)
        blended_rmse = rmse(val.rating, blended_preds)

        sweep_results.append({
            "hazard": hazard,
            "blended_rmse": blended_rmse,
            "improvement_pct": (baseline_rmse - blended_rmse) / baseline_rmse * 100,
        })

    best = min(sweep_results, key=lambda x: x["blended_rmse"])

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"bocpd_sweep_{dataset}_{split}.json", "w") as f:
        json.dump({"baseline_rmse": baseline_rmse, "sweep": sweep_results}, f, indent=2)

    return best["hazard"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movielens", choices=list(RATING_SCALE))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--results-dir", default="results/bocpd_blending")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--hazard", type=float, default=1/50)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        best_hazard = run_sweep(args.dataset, args.split, args.results_dir)
        run_blending_experiment(args.dataset, args.split, args.results_dir, args.factors, best_hazard)
    else:
        run_blending_experiment(args.dataset, args.split, args.results_dir, args.factors, args.hazard)
