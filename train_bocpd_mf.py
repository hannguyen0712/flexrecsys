import argparse
import json
from pathlib import Path

from bocpd import compute_residuals, detect_user_changepoints, generate_weights
from mf import MF
from data import load_split, set_seed, RATING_SCALE
from metrics import rmse

# Train MF with binary BOCPD weights: down-weight pre-changepoint ratings.
def run_binary_weighting(dataset, split, results_dir, n_factors=64, sweep=False):
    set_seed()
    scale = RATING_SCALE[dataset]
    train, val, _ = load_split(dataset, split)

    # baseline: standard MF without weighting
    mf_initial = MF(n_factors=n_factors, rating_scale=scale, verbose=False).fit(train, val)
    baseline_rmse = rmse(val.rating, mf_initial.predict(val))

    # detect changepoints from prediction residuals
    residuals = compute_residuals(mf_initial, train)
    changepoints = detect_user_changepoints(residuals)
    n_users_with_cp = sum(1 for cp in changepoints.values() if cp)

    # weighted MF: 0.5 for old ratings, 1.0 for post-changepoint
    weights = generate_weights(train, changepoints, base_weight=0.5)
    mf_weighted = MF(n_factors=n_factors, rating_scale=scale, verbose=False).fit(train, val, weights=weights)
    weighted_rmse = rmse(val.rating, mf_weighted.predict(val))

    results = {
        "dataset": dataset,
        "split": split,
        "method": "binary_weighting",
        "n_factors": n_factors,
        "baseline_rmse": float(baseline_rmse),
        "weighted_rmse": float(weighted_rmse),
        "improvement_pct": float((baseline_rmse - weighted_rmse) / baseline_rmse * 100),
        "n_users": len(changepoints),
        "users_with_changepoints": n_users_with_cp,
        "best_base_weight": 0.5,
        "best_hazard": 1/50,
    }

    # tuning the hazard parameter
    if sweep:
        base_results = []
        for base in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            w = generate_weights(train, changepoints, base_weight=base)
            model = MF(n_factors=n_factors, rating_scale=scale, epochs=15, verbose=False).fit(train, val, weights=w)
            val_rmse = rmse(val.rating, model.predict(val))
            base_results.append({"base_weight": base, "rmse": val_rmse})

        best_base = min(base_results, key=lambda x: x["rmse"])["base_weight"]

        hazard_results = []
        for hazard in [1/20, 1/30, 1/50, 1/75, 1/100]:
            cp = detect_user_changepoints(residuals, hazard=hazard)
            n_cp = sum(1 for c in cp.values() if c)
            w = generate_weights(train, cp, base_weight=best_base)
            model = MF(n_factors=n_factors, rating_scale=scale, epochs=15, verbose=False).fit(train, val, weights=w)
            val_rmse = rmse(val.rating, model.predict(val))
            hazard_results.append({"hazard": hazard, "rmse": val_rmse, "users_with_cp": n_cp})

        best_hazard_result = min(hazard_results, key=lambda x: x["rmse"])
        results["sweep"] = {"base_weight_results": base_results, "hazard_results": hazard_results}
        results["best_base_weight"] = best_base
        results["best_hazard"] = best_hazard_result["hazard"]
        results["best_rmse"] = best_hazard_result["rmse"]
        results["best_improvement_pct"] = float((baseline_rmse - best_hazard_result["rmse"]) / baseline_rmse * 100)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"bocpd_binary_{dataset}_{split}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movielens", choices=list(RATING_SCALE))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--results-dir", default="results/bocpd_binary")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    run_binary_weighting(args.dataset, args.split, args.results_dir, args.factors, args.sweep)
