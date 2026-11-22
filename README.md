# flexrecsys

Detecting preference shifts in matrix factorization recommenders.

Most time-aware recommenders assume user tastes drift smoothly and down-weight old ratings with a continuous decay. With flexrecsys, we instead treat preference changes as discrete events. The system locates a change point in each user's rating history with a Bayesian test and re-estimates the user from the ratings that follow it.
sys-level` | Keeps the full-history taste vector, re-estimates only the user bias from post-change ratings |

The system has three stages:

1. **Train matrix factorization** with weight decay.
2. **Detect change points**: for each user, the probability that their taste changed, and where.
3. **Re-solve each user exactly** with the movie vectors frozen, using the change points to decide how much each rating counts.

Without stage 2, every rating counts fully under flexrecsys (mf-refit only) model. The other models differ in how stage 2 is done or how its output is used; see [The flexrecsys models](#the-flexrecsys-models).

## Summary of results

1. **flexrecsys out-ranks TimeSVD++ on two datasets, without its change-point stage.** flexrecsys (mf-refit only) beats TimeSVD++ on held-out test Recall@10 by 20% on MovieLens (0.0742 against 0.0616, paired difference +0.0126, 95% interval +0.0084 to +0.0166) and by 26% on Netflix (0.0622 against 0.0495, paired difference +0.0128, interval +0.0087 to +0.0168). It trains in about 6 seconds against 140. TimeSVD++ keeps the better RMSE on both.
2. **The original gap to TimeSVD++ was a regularization artifact.** Switching off every time-aware component of TimeSVD++ leaves its ranking unchanged. Training plain MF with the same regularization closes the gap.
3. **The change-point stage does not help on MovieLens or Netflix.** Three detectors were tried: online BOCPD on embedding windows (the proposal's method), online BOCPD on prediction errors, and an offline rating-space test. No variant gives a confirmed improvement over flexrecsys (mf-refit only). The only candidate is a 0.15% validation RMSE gain for one setting of flexrecsys (residual BOCPD, binary weights), chosen from a sweep and not yet checked on held-out data or on ranking. Discarding pre-change ratings makes flagged users worse on both datasets.
4. **The detector finds real changes, but they are gradual and do not last.** It flags 35% of MovieLens users against 5% when each user's rating order is shuffled. On MovieLens, flagged users' post-change ratings predict their future better than their pre-change ratings, and their genre mix moves more than unflagged users'. The moves are shifts in the balance of common genres, and a user's full history still predicts their future better than the post-change history does.
5. **flexrecsys (soft update) avoids the loss but does not gain.** It chose strong pulls toward each user's full-history profile (x30 to x300) and matched flexrecsys (mf-refit only), including on the Netflix test split (0.0621 against 0.0622 Recall@10). The strength is chosen on validation data, and this variant has not been scored on the MovieLens test split.

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/hannguyen0712/flexrecsys.git
cd flexrecsys
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Conda users can run `conda env create -f environment.yml` instead.

## Data

Two public datasets are supported. Neither is included in this repository.

```bash
python data.py movielens         # MovieLens 25M, downloaded from GroupLens
python data.py netflix           # Netflix Prize, downloaded from Kaggle
```

The Netflix Prize data requires a Kaggle account and API token (`~/.kaggle/kaggle.json`). It contains about 100M ratings; use `--dev-only` on machines with less than 16 GB of RAM.

Users need at least 50 ratings spanning 180 days, movies at least 5 ratings. Each user's ratings are split chronologically 80/10/10 into train, validation and test. The `dev` split is a fixed 10% sample of users. All MovieLens results below use it: 851,711 train, 105,443 validation and 108,757 test ratings from 2,591 users and 20,634 movies.

## Method

**Stage 1, matrix factorization.** `r = mu + b_u + b_i + p_u . q_i`, trained with Adam weight decay (32 factors, learning rate 0.003, 8 epochs, weight decay 1e-4).

**Stage 3, user refit.** With the movie factors frozen, each user's vector and bias are re-solved in closed form by weighted ridge regression on their training ratings. The ridge strength is chosen per seed on validation NDCG@10 from a fixed grid, using the refit with all weights 1, and is then shared by every variant that refits.

**Stage 2, change-point detection.** Three detectors are implemented:

- **Window BOCPD**, the method in the proposal: each user's history is cut into windows of 10 ratings, a user vector is fitted per window, and Bayesian online change-point detection (Adams and MacKay) runs over the window vectors.
- **Residual BOCPD** (`bocpd.py`): online change-point detection on each user's prediction errors from the trained MF. See [BOCPD_README.md](BOCPD_README.md).
- **Rating-space test**, the final detector: for each user, with movies frozen, it compares the evidence for one user vector over the whole history against two vectors split at each candidate point (at least 10 ratings per side, prior probability of a change 0.2). The output is the probability of a change and, for every rating, the probability that it belongs to the user's current regime. A user is called flagged when the change probability exceeds 0.5. By default the detector's prior strength and noise level are fitted to the training ratings (`--regime-prior auto`); `--regime-prior fixed` uses `--cp-reg` and `--cp-noise`.

### The flexrecsys models

Every flexrecsys model starts from the same trained matrix factorization (stage 1). They differ in two ways: which detector, if any, looks for a change in each user's history (stage 2), and what is done with what it finds. The table is a quick reference; each model is explained below it.

| Name in the tables | `--models` key | Detector | What it does with a detected change |
|---|---|---|---|
| flexrecsys (mf-refit only) | `mf-refit` | none | Nothing: every rating counts fully |
| flexrecsys (hard update) | `flexrecsys` | rating-space test | Rebuilds flagged users from their post-change ratings only |
| flexrecsys (soft update) | `flexrecsys-soft` | rating-space test | Lets flagged users move part of the way toward their post-change ratings |
| flexrecsys (level update) | `flexrecsys-level` | rating-space test | Updates only how generously flagged users rate, not their taste |
| flexrecsys (window BOCPD) | `flexrecsys-v1` | window BOCPD | Down-weights earlier ratings and retrains the whole model |
| flexrecsys (residual BOCPD, binary weights) | `train_bocpd_mf.py` | residual BOCPD | Down-weights ratings before the last change and retrains the whole model |
| flexrecsys (residual BOCPD, blending) | `train_bocpd_blending.py` | residual BOCPD | Mixes a long-term and a short-term profile for each user |

**flexrecsys (mf-refit only)** uses no change points. After the matrix factorization is trained, each user's profile (their taste vector and rating bias) is recomputed exactly from all of their training ratings, with the movie vectors held fixed. Eight epochs of gradient descent leave user profiles undertrained, and this exact solve recovers the personalization. Because it is stages 1 and 3 with every rating counting fully, it is the reference for every other variant: any difference from it is the effect of change points. It is the best-ranking model in this project.

**flexrecsys (hard update)** is the main change-point model. The rating-space test estimates, for every rating, the probability that it belongs to the user's current taste. The refit then weights each rating by that probability. For a user the detector is confident about, ratings before the change count for almost nothing, so the profile is rebuilt from post-change ratings only; users with no detected change are left as they are. It assumes that a detected change is complete and lasting.

**flexrecsys (soft update)** uses the same detector, but treats a change as partial. Each flagged user starts from their full-history profile and moves toward what their post-change ratings say, held back by a pull toward the full-history profile. The strength of that pull is chosen on validation data: at the weakest setting it behaves like the hard update, and at the strongest it barely moves, like flexrecsys (mf-refit only). Because the strength is chosen on validation data, it should not do much worse than flexrecsys (mf-refit only) unless that choice fails to carry over to new data.

**flexrecsys (level update)** uses the same detector, but keeps each user's taste vector from their full history and re-estimates only their rating bias from post-change ratings. It tests the idea that part of a detected change is a user becoming a more generous or harsher rater. Raising or lowering all of one user's scores by the same amount cannot reorder their recommendations, so its ranking is identical to flexrecsys (mf-refit only) by construction; it can only change rating accuracy.

**flexrecsys (window BOCPD)** is the method in the proposal. Each user's history is cut into windows of 10 ratings, a profile is fitted to each window, and Bayesian online change-point detection runs over the sequence of window profiles. Its run-length posterior gives each rating a weight, so ratings from earlier regimes count for as little as 0.2, and the whole matrix factorization, movie vectors included, is retrained with those weights. It has no stage 3 refit, so the model to compare it with is plain MF.

**flexrecsys (residual BOCPD, binary weights)** runs Bayesian online change-point detection on each user's sequence of prediction errors from the trained matrix factorization: if a user's errors shift, the model has stopped fitting them. A change is recorded where the most likely run length drops sharply. Ratings before the user's last change get a fixed lower weight (0.5 by default), and the whole model is retrained. These runs use the original 64-factor recipe and report RMSE only.

**flexrecsys (residual BOCPD, blending)** uses the same detector but retrains nothing. For each user it fits a short-term profile to their ratings after the last detected change (or their most recent 20% of ratings if no change is found) and mixes it with their long-term profile. The mixing weight is the expected run length divided by the number of ratings, so users who look stable lean on their long-term profile.

**Comparison methods.** Plain MF is stage 1 alone. The bias baseline predicts from the overall average plus user and movie biases. TimeSVD++ is a time-aware matrix factorization (see [TIMESVDPP_README.md](TIMESVDPP_README.md)). "Time decay, users refit" (`decay-refit`) is the continuous-decay approach the proposal compares against: the same exact refit as flexrecsys (mf-refit only), but with every rating's weight halving for each 180 days between it and the user's latest rating, so old ratings fade smoothly instead of being cut off at a change point.

## Usage

`compare.py` defaults to the final configuration: MovieLens dev, the weight-decay recipe above, and seeds 575, 576 and 577. So most runs only name the models.

```bash
python compare.py --models mf mf-refit flexrecsys                       # MovieLens, validation
python compare.py --data netflix_day_dev --models mf mf-refit flexrecsys # Netflix 1% sample
```

The test tables below were produced with:

```bash
python compare.py --eval test --models bias mf mf-refit decay-refit timesvdpp flexrecsys --regime-prior fixed --tag final_test
python compare.py --data netflix_day_dev --eval test --models bias mf mf-refit decay-refit timesvdpp flexrecsys flexrecsys-soft flexrecsys-level --tag netflix_test
python compare.py --eval test --recipe old --models mf --tag final_test_oldmf
```

Each run writes `results/<tag>.json`, per-user scores, a figure (PNG and PDF) and tables (Markdown, LaTeX, CSV). When a variant with a detector is included it also writes `<tag>_flags.csv` and, on the first seed, runs the flag checks reported below (placebo, before/after, and the genre shift on MovieLens), saved under `diagnostics` in the JSON. `--no-diagnose` skips them.

```bash
python subgroup.py results/final_test     # hard update vs mf-refit only: flagged, unflagged, by history length
python plots.py results/final_test.json   # redraw figure and tables
```

- `--eval test` scores the same train-only models on test; every choice (early stopping, refit strength, TimeSVD++ epoch count) is still made on validation. The flag checks always use validation.
- `--recipe old` restores the original per-batch L2 training settings; single settings such as `--factors` can still be set on their own.
- `--fresh-users N` draws users from the full split that are not in dev.
- Run `python compare.py --help` for all options. The standalone TimeSVD++ pipelines are described in [TIMESVDPP_README.md](TIMESVDPP_README.md), and the residual BOCPD scripts in [BOCPD_README.md](BOCPD_README.md).

## Results

### MovieLens, test split

Three seeds (575, 576, 577). The configuration was frozen before this run, and the test split was scored once.

| Model | RMSE ↓ | Recall@10 ↑ | Recall@20 ↑ | NDCG@10 ↑ | NDCG@20 ↑ |
|:---|---:|---:|---:|---:|---:|
| Bias baseline | 0.8444 | 0.0346 ± 0.0031 | 0.0444 ± 0.0035 | 0.0368 ± 0.0033 | 0.0407 ± 0.0031 |
| MF | 0.8357 ± 0.0001 | 0.0588 ± 0.0007 | 0.0652 ± 0.0005 | 0.0612 ± 0.0004 | 0.0619 ± 0.0004 |
| Time decay, users refit | 0.8220 ± 0.0011 | 0.0722 ± 0.0008 | 0.0792 ± 0.0003 | 0.0757 ± 0.0009 | 0.0754 ± 0.0006 |
| TimeSVD++ | **0.8150 ± 0.0007** | 0.0616 ± 0.0055 | 0.0693 ± 0.0029 | 0.0643 ± 0.0068 | 0.0651 ± 0.0050 |
| flexrecsys (mf-refit only) | 0.8225 ± 0.0000 | **0.0742 ± 0.0006** | **0.0816 ± 0.0009** | **0.0783 ± 0.0004** | **0.0779 ± 0.0008** |
| flexrecsys (hard update) | 0.8408 ± 0.0007 | 0.0715 ± 0.0006 | 0.0792 ± 0.0010 | 0.0755 ± 0.0009 | 0.0755 ± 0.0010 |

Intervals are ±1 standard deviation over seeds, except the bias baseline (deterministic, 95% bootstrap interval over users). Ranking metrics cover the 2,571 users with at least one test rating of 4.0 or higher. flexrecsys (hard update) here uses the fixed detector prior (`--cp-reg 5`).

![Test metrics](results/final_test.png)

- flexrecsys (mf-refit only) beats TimeSVD++ on ranking on every seed; its lowest Recall@10 (0.0738) is above TimeSVD++'s highest (0.0665).
- TimeSVD++ has the best rating accuracy in this table.
- flexrecsys (hard update) is below flexrecsys (mf-refit only) on Recall@10 by 0.0028 (95% interval 0.0016 to 0.0041). The loss is confined to the 807 flagged users (Recall@10 down 0.0087, squared error up 0.077); the 1,764 unflagged users are unchanged.
- Time decay does not improve on the unweighted refit either.
- Longer histories do not help. flexrecsys (hard update) loses Recall@10 for flagged users with short (40 to 142 ratings), medium (143 to 316) and long (317 or more) histories: by 0.0035 (interval −0.0103 to +0.0024), 0.0128 (−0.0181 to −0.0079) and 0.0081 (−0.0142 to −0.0020).

### Netflix, test split

A 1% sample of Netflix Prize users with day-aware splits (`data/netflix_day_dev`): 641,771 train, 72,023 validation and 68,394 test ratings from 2,393 users and 14,453 movies. The configuration frozen on MovieLens was run unchanged, on the 1 to 5 rating scale and with the fitted detector prior. Three seeds, test scored once.

| Model | RMSE ↓ | Recall@10 ↑ | Recall@20 ↑ | NDCG@10 ↑ | NDCG@20 ↑ |
|:---|---:|---:|---:|---:|---:|
| Bias baseline | 0.9225 | 0.0421 ± 0.0044 | 0.0427 ± 0.0040 | 0.0461 ± 0.0049 | 0.0421 ± 0.0039 |
| MF | 0.9079 ± 0.0001 | 0.0550 ± 0.0003 | 0.0554 ± 0.0007 | 0.0591 ± 0.0005 | 0.0549 ± 0.0006 |
| Time decay, users refit | 0.8989 ± 0.0014 | 0.0622 ± 0.0009 | 0.0671 ± 0.0006 | **0.0648 ± 0.0006** | **0.0631 ± 0.0005** |
| TimeSVD++ | **0.8857 ± 0.0008** | 0.0495 ± 0.0033 | 0.0574 ± 0.0028 | 0.0493 ± 0.0044 | 0.0507 ± 0.0029 |
| flexrecsys (mf-refit only) | 0.9031 ± 0.0014 | **0.0622 ± 0.0013** | 0.0672 ± 0.0002 | 0.0644 ± 0.0005 | 0.0628 ± 0.0001 |
| flexrecsys (hard update) | 0.9069 ± 0.0017 | 0.0613 ± 0.0015 | 0.0664 ± 0.0002 | 0.0635 ± 0.0005 | 0.0620 ± 0.0004 |
| flexrecsys (soft update) | 0.9019 ± 0.0017 | 0.0621 ± 0.0013 | **0.0674 ± 0.0002** | 0.0643 ± 0.0006 | 0.0629 ± 0.0001 |
| flexrecsys (level update) | 0.9051 ± 0.0015 | 0.0622 ± 0.0013 | 0.0672 ± 0.0002 | 0.0644 ± 0.0005 | 0.0628 ± 0.0001 |

Ranking metrics cover the 2,310 users with at least one test rating of 4.0 or higher.

- flexrecsys (mf-refit only) beats TimeSVD++ on Recall@10 by 0.0128 (95% interval 0.0087 to 0.0168), and TimeSVD++ has the best RMSE, as on MovieLens.
- For the 561 flagged users, against flexrecsys (mf-refit only): the hard update lowers Recall@10 by 0.0058 (interval 0.0013 to 0.0104) and raises squared error by 0.052 (0.021 to 0.087); the soft update has no detectable effect; the level update raises squared error by 0.025 (0.003 to 0.051).
- The detector flags 23.8% of users and 5.7% after shuffling each user's rating order. Unlike MovieLens, the before/after check does not separate the groups: the "after" profile is better by 0.113 squared error for flagged users (interval −0.010 to +0.225) and by 0.072 for unflagged users (0.043 to 0.102).

### Why the first comparison was misleading

The project's first results had TimeSVD++ ranking four to five times better than plain MF. That gap came from how the two models were regularized.

| Model, validation, seed 575 | RMSE | Recall@10 |
|:---|---:|---:|
| MF, per-batch L2 penalty (original) | 0.7821 | 0.0115 |
| MF, weight decay only, otherwise original settings | 0.8127 | 0.0543 |
| TimeSVD++, full | 0.7820 | 0.0585 |
| TimeSVD++, no movie bias by period | 0.7812 | 0.0562 |
| TimeSVD++, no user drift | 0.7849 | 0.0593 |
| TimeSVD++, no user-day effects | 0.7827 | 0.0591 |
| TimeSVD++, no implicit feedback | 0.8083 | 0.0597 |
| TimeSVD++, all four switched off | 0.8130 | 0.0550 |

No component of TimeSVD++ matters for ranking; implicit feedback matters for RMSE. Adam weight decay shrinks every movie's parameters on every step, which keeps rarely rated movies out of the top of the ranking. On test the original MF recipe scores 0.0140 Recall@10 against 0.0588 for the weight-decay recipe.

The two measures pull against each other throughout. The original MF recipe has the best test RMSE of any model (0.8049) and the worst ranking. Training TimeSVD++ for 30 epochs instead of 8 improves its validation RMSE from 0.7820 to 0.7762 and lowers its Recall@10 from 0.0585 to 0.0460.

### The change-point stage on MovieLens

Validation, three seeds, same base model.

| Model | RMSE | Recall@10 | NDCG@10 |
|:---|---:|---:|---:|
| MF | 0.8136 | 0.0531 | 0.0547 |
| Time decay, users refit | 0.7912 | 0.0714 | 0.0735 |
| flexrecsys (mf-refit only) | 0.7948 | 0.0738 | 0.0760 |
| flexrecsys (window BOCPD) | 0.8133 | 0.0489 | 0.0511 |
| flexrecsys (hard update, fixed prior) | 0.8035 | 0.0733 | 0.0759 |
| flexrecsys (hard update, fitted prior) | 0.8051 | 0.0726 | 0.0756 |
| flexrecsys (soft update) | 0.7947 | 0.0741 | 0.0763 |
| flexrecsys (level update) | 0.7990 | 0.0738 | 0.0760 |

flexrecsys (window BOCPD) flags 85% of users and ranks 8% below plain MF, the model it retrains. None of the other variants is distinguishable from flexrecsys (mf-refit only) on ranking. For flagged users, the hard update raises squared error (+0.039, interval +0.014 to +0.064), the soft update has no detectable effect (Recall@10 +0.0011, interval −0.0014 to +0.0041) and chooses a strength that keeps users close to their full-history vector, and the level update raises squared error slightly (+0.018, interval +0.0001 to +0.036).

The flag checks test whether the rating-space detector's flags are meaningful:

- **Placebo.** The detector flags 35.0% of users. After shuffling the order of each user's ratings it flags 5.0%.
- **Before and after.** Fitting a user on the ratings just before the detected change, against the same number just after, the "after" profile predicts validation ratings better by 0.215 squared error for flagged users (interval 0.134 to 0.299) and by 0.090 for unflagged users (0.068 to 0.112).
- **Genre shift.** The genre mix of the ratings just after the change differs from the mix just before it, beyond what random splits of the same ratings give, by 0.050 Jensen-Shannon divergence for flagged users (interval 0.045 to 0.055) and 0.039 for unflagged users (0.036 to 0.043). 44.1% of flagged and 30.9% of unflagged users move beyond chance, where 5% is nominal. The genres that move most are the most common ones (Drama, Comedy, Action, Thriller, Romance).

So the detector finds real changes, partly in what users watch and not only in how they rate. The changes are gradual shifts in the balance of common genres, and they are not durable enough to act on: the full history predicts a user's future better than the post-change history does. Because the comparison point for every user is the detector's best guess at a change, the genre measure is somewhat inflated for both groups; the comparison between them is the informative one.

### Residual BOCPD

Online change-point detection on each user's prediction errors from MF trained with the original recipe (64 factors), the proposal's "likelihood of new interactions" signal. Validation RMSE, one seed; these scripts do not compute ranking metrics.

| MovieLens | RMSE |
|:---|---:|
| MF | 0.7821 |
| flexrecsys (residual BOCPD, binary weights), default settings (pre-change weight 0.5, hazard 1/50) | 0.7825 |
| flexrecsys (residual BOCPD, binary weights), pre-change weight 1.0 (no down-weighting) | 0.7820 |
| flexrecsys (residual BOCPD, binary weights), best of 11 settings (weight 0.75, hazard 1/20) | 0.7809 |
| flexrecsys (residual BOCPD, binary weights), pre-change ratings dropped (weight 0) | 0.8240 |
| flexrecsys (residual BOCPD, blending), hazard 1/100 | 0.7895 |

- Retraining with identical weights moves RMSE by about 0.0002 (MF against weight 1.0; the default settings run twice give 0.7825 and 0.7823).
- The best binary-weights setting is 0.0012 (0.15%) below MF, which is beyond that noise. It was selected from 11 settings on the validation set it is reported on, from one seed, and has not been checked on test data or on ranking. The effect is not monotone in the hazard rate (1/30 and 1/50 are worse than 1/75).
- Down-weighting pre-change ratings more strongly than 0.5 makes RMSE clearly worse.
- Blending is worse than MF on both datasets (Netflix: 0.8536 against 0.8520). It blends every user, including those with no detected change, and it mixes a vector trained by SGD with one fitted by ridge regression, so part of the loss may come from that mismatch rather than from the shift.
- An earlier version of `generate_weights` gave users with no detected change the pre-change weight on all their ratings; the MovieLens numbers above come from the corrected code. The Netflix binary-weights sweep (24,582 users) has not been rerun with the fix.

### Implementation check on generated data

To check that the detector finds what it is designed to find, `synthetic.py` generates ratings from a matrix factorization in which each user has one favourite genre and half the users switch, once and permanently, to another genre partway through their history. This is a check of the implementation, not evidence about real users: the data is generated by the same kind of model the system fits, and the switches match the detector's assumption of at most one change.

On this data the rating-space detector recovers the planted switches (AUC 0.82, precision 0.83, three seeds). With no planted switches it flags 4.3% of users, about the same as on shuffled histories (4.4%).

```bash
python compare.py --synthetic --models mf-refit flexrecsys
```

## Limitations

- Small samples of two datasets: 10% of eligible MovieLens users and 1% of Netflix users.
- Netflix records dates, not times, so ratings made on the same day have no true order; ties are broken by movie ID.
- The strength of flexrecsys (soft update) is chosen on the same validation data it is scored on.
- There is no ground truth for preference shifts in either dataset, so on real data the detector is checked indirectly (placebo, before/after, genre shift). It is checked against known changes only on generated data.
- Many variants were tried against the same MovieLens validation data. Only the frozen configuration in the MovieLens test table was scored on that test split. The fitted detector prior and the soft and level updates were developed after that run; on MovieLens they have validation results only, and their held-out results come from Netflix.
- The refit strength is chosen for ranking. Choosing it for RMSE gives better rating accuracy and worse ranking.
- The TimeSVD++ implementation is an unverified prototype with default hyperparameters. Its epoch count was not tuned for ranking.
- The rating-space detector allows at most one change per user.
- The flag checks cannot separate a change in rating level from a change in taste; the two are entangled in the user profile. The genre shift covers taste only on MovieLens, since Netflix has no genre data.
- The residual BOCPD results are single-seed, RMSE-only and use the original MF recipe, so they are not directly comparable with the `compare.py` tables.
- A Surprise-based MF baseline and a simple threshold detector, both in the proposal, were not implemented.

## Project structure

| File | Description |
|---|---|
| `data.py` | Dataset download, filtering and chronological splits |
| `mf.py` | Bias baseline, PyTorch matrix factorization, closed-form user refits |
| `changepoint.py` | Rating-space detector, fitted prior, time-decay weights, window BOCPD, and the flag checks |
| `bocpd.py` | Residual BOCPD, binary weights and blending |
| `train_bocpd_mf.py`, `train_bocpd_blending.py` | flexrecsys (residual BOCPD, binary weights) and (residual BOCPD, blending), with sweeps |
| `metrics.py` | RMSE, Recall@K, NDCG@K |
| `compare.py` | Runs all models on one split and protocol, saves results |
| `subgroup.py` | Paired comparison of two models for flagged and unflagged users, and by history length |
| `synthetic.py` | Generated ratings with planted genre switches, used only to check the detector |
| `plots.py` | Comparison figure and tables from a results file |
| `tune_mf.py`, `plot_results.py` | Validation tuning of the original MF recipe and its figures |
| `timesvdpp_full.py` | TimeSVD++-style model and training, with switchable components |
| `evaluate_timesvdpp_ranking.py` | Ranking evaluation for TimeSVD++ |
| `timesvdpp_movielens*.py`, `evaluate_timesvdpp_movielens*.py` | Standalone TimeSVD++ runs on MovieLens, standard and day-aware splits |
| `movielens_day_split.py` | Day-aware chronological splits for MovieLens |
| `preprocess_netflix.py` | Streams the raw Netflix files into a 1% user sample |
| `netflix_day_split.py` | Day-aware chronological splits for Netflix |
| `han-sandbox.ipynb` | Exploration: PCA of item factors against genres |
| `results/` | Saved metrics, per-user scores, figures and tables |

## License

Apache 2.0. See [LICENSE](LICENSE).
