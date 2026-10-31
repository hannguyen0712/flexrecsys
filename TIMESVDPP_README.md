# Netflix TimeSVD++ Baseline Experiment

## Overview

This experiment implements a TimeSVD++-style temporal matrix factorization recommender using the Netflix Prize dataset.

The model includes temporal user biases, time-varying user factors, item time-bin biases, user-day effects, and SVD++ implicit-feedback factors.

**Status:** Experimental prototype. This is not yet a verified reproduction of the original published TimeSVD++ implementation.

## Dataset

Download the Netflix Prize dataset from:

https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data

The dataset must include `combined_data_1.txt` through `combined_data_4.txt`.

## Reproduction steps

Run the following commands from the repository root after installing the dependencies.

### 1. Preprocess Netflix

```bash
python preprocess_netflix.py --netflix-dir "/path/to/netflix/files"
```

### 2. Generate day-aware chronological splits

```bash
python netflix_day_split.py
```

The resulting files are stored in `data/netflix_day_dev/`.

### 3. Train TimeSVD++

```bash
python timesvdpp_full.py --epochs 8
```

The best checkpoint is saved to `results/timesvdpp_netflix_best.pt`.

### 4. Evaluate recommendation rankings

```bash
python evaluate_timesvdpp_ranking.py
```

## Validation results

| Metric | Result |
|---|---:|
| RMSE | 0.86226 |
| Recall@10 | 0.05623 |
| Recall@20 | 0.06268 |
| NDCG@10 | 0.05607 |
| NDCG@20 | 0.05644 |

Training used 641,771 ratings from 2,393 users. Validation contained 72,023 ratings.

The model was trained for eight epochs with seed 575.

## Evaluation considerations

- Netflix provides day-level timestamps.
- The day-aware split prevents same-day train/validation overlap.
- Ranking uses the training movie catalog, excluding movies already rated by each user in training.
- Ranking is evaluated at each user's first validation day.
- Relevance is defined as a validation rating of at least 4.0.
- Unseen validation movies receive a fallback prediction for RMSE and cannot be recommended in the ranking evaluation.
- The test set has not been used for model selection.
- Comparisons against other recommenders must use the same data split and evaluation protocol.

## Limitations

The implementation uses complete training-only user histories for SVD++ implicit-feedback aggregation rather than strictly updating histories after each interaction.

The model remains an experimental TimeSVD++-style implementation, and its correspondence to the published formulation requires further validation.