# BOCPD for Recommender Systems

Bayesian Online Changepoint Detection (BOCPD) for detecting user preference shifts in collaborative filtering.

## Overview

Users' preferences change over time. This module uses BOCPD to detect when preferences shift and adjusts recommendations accordingly.

### Two Approaches

| Approach | Description | File |
|----------|-------------|------|
| **Binary Weighting** | Down-weight ratings before detected changepoint | `train_bocpd_mf.py` |
| **Embedding Blending** | Blend long-term and short-term user embeddings using run length posterior | `train_bocpd_blending.py` |

## How BOCPD Works

BOCPD maintains a probability distribution over "run length" — how many observations since the last changepoint.

```
User ratings:  [4, 5, 4, 5, 2, 1, 2, 1]
                        ↑
                   Preference shift detected

Run length:    [1, 2, 3, 4, 1, 2, 3, 4]
                        ↑
                   Resets after changepoint
```

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

### Experimental Findings

| Dataset | Baseline | Binary | Blending |
|---------|----------|--------|----------|
| MovieLens | 0.7821 | TBD | 0.7943 (-1.56%) |
| Netflix | 0.8520 | TBD | 0.8604 (-0.98%) |

### Why Negative Results?

The embedding blending approach performed worse than baseline. Analysis:

1. **Most users are stable**: ~50% have α > 0.7, meaning they lean heavily toward long-term preferences
2. **Short-term embeddings are noisy**: Ridge regression with few ratings produces unreliable vectors
3. **Dataset characteristics**: MovieLens and Netflix contain mostly stable users with gradual preference evolution, not abrupt shifts





