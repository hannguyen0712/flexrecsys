# BOCPD for Recommender Systems

Bayesian Online Changepoint Detection (BOCPD) for detecting user preference shifts in collaborative filtering.

## Overview

Users' preferences change over time. This module uses BOCPD to detect when preferences shift and adjusts recommendations accordingly.

### Two Approaches

Both are flexrecsys variants and are named that way in the main [README](README.md).

| Name | Description | File |
|----------|-------------|------|
| **flexrecsys (residual BOCPD, binary weights)** | Down-weight ratings before detected changepoint | `train_bocpd_mf.py` |
| **flexrecsys (residual BOCPD, blending)** | Blend long-term and short-term user embeddings using run length posterior | `train_bocpd_blending.py` |

## How BOCPD Works

BOCPD maintains a probability distribution over "run length" — how many observations since the last changepoint.


### Key Concept: Run Length Posterior

At each time step, BOCPD computes `P(run_length | data)`:
- **High expected run length** → User is stable → Trust historical preferences
- **Low expected run length** → Recent shift → Trust recent behavior

### Blending Formula
```
α = E[run_length] / T          # α ∈ [0, 1]
p_final = α·p_long + (1-α)·p_short
```

Where:
- `p_long`: User embedding from all ratings
- `p_short`: User embedding from post-changepoint ratings
- `α`: Blending weight from BOCPD

## Quick Start

### Run Binary Weighting
```bash
# MovieLens
python train_bocpd_mf.py --dataset movielens --split dev

# Netflix
python train_bocpd_mf.py --dataset netflix --split dev

# With hyperparameter sweep
python train_bocpd_mf.py --dataset movielens --sweep
```

### Run Embedding Blending
```bash
# MovieLens
python train_bocpd_blending.py --dataset movielens --split dev

# Netflix
python train_bocpd_blending.py --dataset netflix --split dev

# With hyperparameter sweep
python train_bocpd_blending.py --dataset movielens --sweep
```

## Results

Results are saved to `results/` as JSON files:

```
results/
├── bocpd_binary/
│   ├── bocpd_binary_movielens_dev.json
│   └── bocpd_binary_netflix_dev.json
└── bocpd_blending/
    ├── bocpd_blending_movielens_dev.json
    ├── bocpd_blending_netflix_dev.json
    └── bocpd_sweep_*.json
```

### Sample Output

```json
{
  "baseline_rmse": 0.7821,
  "blended_rmse": 0.7943,
  "improvement_pct": -1.56,
  "users_with_changepoints": 793,
  "alpha_mean": 0.622,
  "alpha_gt_0.7": 1210
}
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `baseline_rmse` | Standard MF without BOCPD |
| `blended_rmse` / `weighted_rmse` | RMSE with BOCPD approach |
| `improvement_pct` | Relative improvement (positive = better) |
| `users_with_changepoints` | Users where preference shift detected |
| `alpha_mean` | Average blending weight (high = stable users) |

## Results Discussion

Validation RMSE, one seed, MF with 64 factors. These scripts do not compute ranking metrics.

### flexrecsys (residual BOCPD, binary weights): MovieLens, corrected `generate_weights`

| Setting | RMSE |
|---------|------|
| MF baseline | 0.7821 |
| Default (pre-change weight 0.5, hazard 1/50) | 0.7825 |
| Pre-change weight 1.0 (no down-weighting) | 0.7820 |
| Best of sweep (weight 0.75, hazard 1/20, 1,456 users with a change) | 0.7809 (+0.15%) |
| Pre-change weight 0 | 0.8240 |

- Retraining with identical weights moves RMSE by about 0.0002, so the best setting's gain of 0.0012 is beyond retraining noise.
- It is the best of 11 settings, chosen on the validation set it is reported on, from one seed, and it has not been checked on the test split or on ranking.
- Stronger down-weighting is clearly worse.

### flexrecsys (residual BOCPD, binary weights): Netflix, corrected `generate_weights`

| Setting | RMSE |
|---------|------|
| MF baseline | 0.8520 |
| Default (pre-change weight 0.5, hazard 1/50) | 0.8560 (-0.47%) |
| Pre-change weight 0.75 | 0.8530 |
| Pre-change weight 1.0 (no down-weighting) | 0.8519 |
| Pre-change weight 0 | 0.8925 |

- RMSE gets steadily worse as pre-change ratings are down-weighted more, so the MovieLens gain does not replicate on Netflix.
- The sweep picks weight 1.0 as best, and the hazard sweep then runs at that weight, where every rating keeps weight 1 whatever the hazard. Its spread (0.8518 to 0.8523) is therefore retraining noise, and its reported best improvement (+0.02%) is not a gain.

An earlier version of `generate_weights` gave users with no detected change the pre-change weight on every rating. Both binary-weighting tables use the corrected function; the earlier Netflix result (0.8508, +0.14%) came from the old version.

### flexrecsys (residual BOCPD, blending)

| Dataset | Baseline | Blending (hazard 1/100) |
|---------|----------|-------------------------|
| MovieLens | 0.7821 | 0.7895 (-0.95%) |
| Netflix | 0.8520 | 0.8536 (-0.19%) |

Every user is blended, including users with no detected change (their last 20% of ratings serves as the short-term window), and the long-term vector comes from SGD while the short-term one comes from ridge regression. Part of the loss may come from that mismatch rather than from preference shifts.

### Key Observations

1. **Most users are stable**: the blending weight averages 0.75 on MovieLens and 0.77 on Netflix, so most users lean heavily toward their long-term vector.
2. **Short-term embeddings are noisy**: ridge regression on few ratings gives unreliable vectors.
3. **Strong down-weighting hurts**: dropping or heavily discounting pre-change ratings makes RMSE worse, as it does for the rating-space detector in `compare.py`.
4. **Dataset characteristics**: MovieLens and Netflix users mostly show gradual preference evolution, not abrupt shifts. See the main [README](README.md) for the ranking results and the flag checks.