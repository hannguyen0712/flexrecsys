# flexrecsys

Detecting abrupt preference shifts in matrix factorization recommenders.

Most time-aware recommenders assume user tastes drift smoothly and down-weight old ratings with a continuous decay. flexrecsys instead treats preference changes as discrete events: it uses Bayesian online change-point detection on signals derived from user embeddings to find when a user's taste shifts, then weights their history accordingly.

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

Two public datasets are supported that are not included in this repository.

```bash
python data.py movielens         # MovieLens 25M, downloaded from GroupLens
python data.py netflix           # Netflix Prize, downloaded from Kaggle
```

The Netflix Prize data requires a Kaggle account and API token (`~/.kaggle/kaggle.json`). It contains about 100M ratings; use `--dev-only` on machines with less than 16 GB of RAM.


## Usage

Train the matrix factorization model and a bias baseline:

```bash
python mf.py --dataset movielens --split dev --factors 64
```

This reports RMSE, Recall@K and NDCG@K on the validation set and saves metrics and learned embeddings to `results/`. Run `python mf.py --help` for all options.


```python
from data import load_split
train, val, test = load_split("movielens", "dev")
```

## Project structure

| File | Description |
|---|---|
| `data.py` | Dataset download, filtering and chronological splits |
| `mf.py` | Bias baseline and PyTorch matrix factorization with per-rating weights |
| `metrics.py` | RMSE, Recall@K, NDCG@K |
