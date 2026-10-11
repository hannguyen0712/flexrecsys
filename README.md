# flexrecsys: shift-aware MF recommender (CSE 575)

## Setup

```bash
git clone <repo-url> flexrecsys && cd flexrecsys

conda env create -f environment.yml && conda activate flexrecsys
# or, without conda (Python 3.11): pip install -r requirements.txt

python data.py      # downloads MovieLens 25M and builds data/dev/ and data/full/
```

`python data.py` prints a hash for each split. Post them in the group chat;
if they match, everyone has identical data.

## Using the shared code

```python
from data import load_split, load_movies, set_seed
from metrics import rmse, evaluate_ranking

set_seed()
train, val, test = load_split("dev")   # use "full" only for final results
```

Run your scripts and notebooks from the repo root so these imports work.

## Notes

- Load data only through `load_split`. Don't re-split on your own.
- Tune on `val`. Use `test` only for the final results tables.
- Report every number through `metrics.py`.
- Settings at the top of `data.py` change only by team agreement.
- Work on a branch, open a pull request, merge after one teammate looks at it.
