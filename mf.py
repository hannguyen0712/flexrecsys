# predicted rating = mu + b_u + b_i + p_u · q_i

# bias baseline mu + b_u + b_i

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import RATING_SCALE, load_split, set_seed
from metrics import evaluate_ranking, rmse

RESULTS_DIR = Path(__file__).parent / "results"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class IdMap:
    # raw userId/movieId <-> contiguous index 0..n-1, built from train only
    def __init__(self, ids):
        self.ids = np.unique(np.asarray(ids))

    def __len__(self):
        return len(self.ids)

    def encode(self, raw):
        # ids not seen in train come back as -1
        raw = np.asarray(raw)
        idx = np.clip(np.searchsorted(self.ids, raw), 0, len(self.ids) - 1)
        return np.where(self.ids[idx] == raw, idx, -1)

    def decode(self, idx):
        return self.ids[np.asarray(idx)]


class BiasBaseline:
    # r_hat = mu + b_u + b_i, regularized averages, no learning loop needed
    def __init__(self, reg_item=10.0, reg_user=15.0, rating_scale=(0.5, 5.0)):
        self.reg_item, self.reg_user, self.rating_scale = reg_item, reg_user, rating_scale

    def fit(self, train):
        self.mu = float(train.rating.mean())
        dev = train.rating - self.mu
        g = dev.groupby(train.movieId)
        self.b_i = g.sum() / (g.size() + self.reg_item)
        resid = dev - train.movieId.map(self.b_i)
        g = resid.groupby(train.userId)
        self.b_u = g.sum() / (g.size() + self.reg_user)
        return self

    def predict(self, df):
        pred = (self.mu
                + df.userId.map(self.b_u).fillna(0).to_numpy()
                + df.movieId.map(self.b_i).fillna(0).to_numpy())
        return np.clip(pred, *self.rating_scale)

    def recommend(self, history, users=None, k=20):
        # mu and b_u are constant per user, so ranking is just b_i minus seen movies
        users = np.unique(history.userId) if users is None else np.asarray(users)
        ranked = self.b_i.sort_values(ascending=False).index.to_numpy()
        seen = history.groupby("userId")["movieId"].apply(set)
        recs = {}
        for u in users:
            s = seen.get(u, set())
            recs[int(u)] = [m for m in ranked[: k + len(s)] if m not in s][:k]
        return recs


class MFNet(nn.Module):
    # r_hat = mu + b_u + b_i + p_u . q_i
    def __init__(self, n_users, n_items, n_factors, mu, init_std=0.01):
        super().__init__()
        self.P = nn.Embedding(n_users, n_factors)
        self.Q = nn.Embedding(n_items, n_factors)
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_items, 1)
        nn.init.normal_(self.P.weight, std=init_std)
        nn.init.normal_(self.Q.weight, std=init_std)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)
        self.register_buffer("mu", torch.tensor(float(mu)))

    def forward(self, u, i):
        p, q = self.P(u), self.Q(i)
        bu, bi = self.b_u(u).squeeze(-1), self.b_i(i).squeeze(-1)
        return self.mu + bu + bi + (p * q).sum(-1), p, q, bu, bi


class MF:
    def __init__(self, n_factors=64, reg=0.05, lr=5e-3, batch_size=4096, epochs=30,
                 patience=3, rating_scale=(0.5, 5.0), device=None, verbose=True,
                 weight_decay=0.0):
        self.n_factors, self.reg, self.lr = n_factors, reg, lr
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.rating_scale = rating_scale
        self.device = device or pick_device()
        self.verbose = verbose
        # Adam weight decay shrinks every row on every step, so rarely rated movies are pulled
        # harder toward zero than with the per-batch l2 (reg), which only touches rows in the batch
        self.weight_decay = weight_decay

    def fit(self, train, val=None, weights=None):
        # weights: one per train row (time decay, change-point downweighting), default all 1
        dev = self.device
        self.users, self.items = IdMap(train.userId), IdMap(train.movieId)
        u = torch.tensor(self.users.encode(train.userId), dtype=torch.long, device=dev)
        i = torch.tensor(self.items.encode(train.movieId), dtype=torch.long, device=dev)
        r = torch.tensor(train.rating.to_numpy(), dtype=torch.float32, device=dev)
        w = (torch.ones_like(r) if weights is None
             else torch.tensor(np.asarray(weights), dtype=torch.float32, device=dev))
        mu = float((w * r).sum() / w.sum())

        self.net = MFNet(len(self.users), len(self.items), self.n_factors, mu).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        self.history = []
        best, best_state, bad = float("inf"), None, 0

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            self.net.train()
            perm = torch.randperm(len(r)).to(dev)
            total = 0.0
            for s in range(0, len(r), self.batch_size):
                b = perm[s : s + self.batch_size]
                pred, p, q, bu, bi = self.net(u[b], i[b])
                # weighted squared error + l2 on the rows this batch touched
                err = (w[b] * (r[b] - pred) ** 2).sum() / w[b].sum()
                l2 = ((p ** 2).sum(-1) + (q ** 2).sum(-1) + bu ** 2 + bi ** 2).mean()
                loss = err + self.reg * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += err.item() * len(b)

            row = {"epoch": epoch, "train_rmse": (total / len(r)) ** 0.5,
                   "seconds": round(time.time() - start, 1)}
            if val is not None:
                row["val_rmse"] = rmse(val.rating, self.predict(val))
            self.history.append(row)
            if self.verbose:
                print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                       for k, v in row.items()))

            # early stopping on val rmse, keep the best epoch's weights
            if val is not None:
                if row["val_rmse"] < best - 1e-4:
                    best, bad = row["val_rmse"], 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, df, batch_size=65536):
        # users or movies not in train get zero vectors and zero bias
        self.net.eval()
        dev = self.device
        u = self.users.encode(df.userId)
        i = self.items.encode(df.movieId)
        out = []
        for s in range(0, len(df), batch_size):
            ub, ib = u[s : s + batch_size], i[s : s + batch_size]
            ku = torch.tensor(ub >= 0, device=dev, dtype=torch.float32)
            ki = torch.tensor(ib >= 0, device=dev, dtype=torch.float32)
            ut = torch.tensor(np.maximum(ub, 0), device=dev)
            it = torch.tensor(np.maximum(ib, 0), device=dev)
            n = self.net
            pred = (n.mu
                    + n.b_u(ut).squeeze(-1) * ku
                    + n.b_i(it).squeeze(-1) * ki
                    + (n.P(ut) * n.Q(it)).sum(-1) * ku * ki)
            out.append(pred.cpu().numpy())
        return np.clip(np.concatenate(out) if out else np.array([]), *self.rating_scale)

    @torch.no_grad()
    def recommend(self, history, users=None, k=20, batch_size=1024):
        # ranks every train movie per user, skipping movies already in their history
        self.net.eval()
        dev = self.device
        users = np.unique(history.userId) if users is None else np.asarray(users)
        uidx = self.users.encode(users)
        users, uidx = users[uidx >= 0], uidx[uidx >= 0]

        hu, hi = self.users.encode(history.userId), self.items.encode(history.movieId)
        keep = (hu >= 0) & (hi >= 0)
        order = np.argsort(hu[keep], kind="stable")
        seen_u, seen_i = hu[keep][order], hi[keep][order]

        Q = self.net.Q.weight
        b_i = self.net.b_i.weight.squeeze(-1)
        recs = {}
        for s in range(0, len(uidx), batch_size):
            b = uidx[s : s + batch_size]
            # mu and b_u don't change a user's ranking, so leave them out
            scores = self.net.P(torch.tensor(b, device=dev)) @ Q.T + b_i
            lo = np.searchsorted(seen_u, b, "left")
            hi_ = np.searchsorted(seen_u, b, "right")
            rows = np.repeat(np.arange(len(b)), hi_ - lo)
            cols = seen_i[np.concatenate([np.arange(a, z) for a, z in zip(lo, hi_)])]
            scores[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)] = -float("inf")
            top = torch.topk(scores, min(k, scores.shape[1]), dim=1).indices.cpu().numpy()
            for user, row in zip(users[s : s + batch_size], top):
                recs[int(user)] = self.items.decode(row).tolist()
        return recs

    def refit_users(self, train, weights=None, reg=1.0, shift=0.0, anchor=None):
        # weighted ridge for p_u and b_u with items frozen, overwrites the user tables.
        # a taste shift changes the user, not the movies, so only the user side sees the weights.
        # anchor (another MF) with shift > 0 adds shift * |p - p_anchor|^2: the user starts from
        # the anchor's vector and moves only as far as the weighted ratings support
        Q = self.net.Q.weight.detach().cpu().numpy().astype(float)
        b_i = self.net.b_i.weight.detach().cpu().numpy().ravel().astype(float)
        u, i = self.users.encode(train.userId), self.items.encode(train.movieId)
        y = train.rating.to_numpy(float) - float(self.net.mu) - b_i[i]
        w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        n, d = len(self.users), Q.shape[1]
        order = np.argsort(u, kind="stable")
        bounds = np.searchsorted(u[order], np.arange(n + 1))
        P, b_u = np.zeros((n, d)), np.zeros(n)
        pull = None
        if anchor is not None and shift > 0:
            pull = shift * np.hstack([anchor.net.P.weight.detach().cpu().numpy().astype(float),
                                      anchor.net.b_u.weight.detach().cpu().numpy().astype(float)])
        for a in range(n):
            rows = order[bounds[a]:bounds[a + 1]]
            X = np.hstack([Q[i[rows]], np.ones((len(rows), 1))])
            Xw = X * w[rows, None]
            rhs = Xw.T @ y[rows]
            strength = reg
            if pull is not None:
                rhs, strength = rhs + pull[a], reg + shift
            sol = np.linalg.solve(X.T @ Xw + strength * np.eye(d + 1), rhs)
            P[a], b_u[a] = sol[:d], sol[d]
        with torch.no_grad():
            self.net.P.weight.copy_(torch.tensor(P, dtype=torch.float32, device=self.device))
            self.net.b_u.weight.copy_(torch.tensor(b_u[:, None], dtype=torch.float32,
                                                   device=self.device))
        return self

    def user_embeddings(self):
        P = self.net.P.weight.detach().cpu().numpy()
        df = pd.DataFrame(P, columns=[f"p{j}" for j in range(P.shape[1])])
        df.insert(0, "b_u", self.net.b_u.weight.detach().cpu().numpy().ravel())
        df.insert(0, "userId", self.users.ids)
        return df

    def item_embeddings(self):
        Q = self.net.Q.weight.detach().cpu().numpy()
        df = pd.DataFrame(Q, columns=[f"q{j}" for j in range(Q.shape[1])])
        df.insert(0, "b_i", self.net.b_i.weight.detach().cpu().numpy().ravel())
        df.insert(0, "movieId", self.items.ids)
        return df

    def user_vectors_from(self, ratings, reg=1.0):
        # refit p_u and b_u from any subset of ratings (e.g. one time window) with movies fixed.
        # closed-form ridge regression per user, used for recent-vs-past embeddings
        Q = self.net.Q.weight.detach().cpu().numpy()
        b_i = self.net.b_i.weight.detach().cpu().numpy().ravel()
        return ridge_user_vectors(ratings, self.items, Q, b_i, float(self.net.mu), reg)


def ridge_user_vectors(ratings, items, Q, b_i, mu, reg=1.0):
    # solves min ||r - mu - b_i - b_u - q_i.p_u||^2 + reg*(||p_u||^2 + b_u^2) per user
    idx = items.encode(ratings.movieId)
    df = ratings.assign(_i=idx)[idx >= 0]
    d = Q.shape[1]
    out_users, out_vecs = [], []
    for user, g in df.groupby("userId"):
        X = np.hstack([Q[g._i.to_numpy()], np.ones((len(g), 1))])
        y = g.rating.to_numpy() - mu - b_i[g._i.to_numpy()]
        out_users.append(user)
        out_vecs.append(np.linalg.solve(X.T @ X + reg * np.eye(d + 1), X.T @ y))
    vecs = np.array(out_vecs).reshape(-1, d + 1)
    res = pd.DataFrame(vecs[:, :d], columns=[f"p{j}" for j in range(d)])
    res.insert(0, "b_u", vecs[:, d])
    res.insert(0, "userId", out_users)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movielens", choices=list(RATING_SCALE))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20])
    args = ap.parse_args()

    set_seed()
    scale = RATING_SCALE[args.dataset]
    train, val, _ = load_split(args.dataset, args.split)
    print(f"{args.dataset}/{args.split}: {len(train):,} train, {len(val):,} val ratings, "
          f"device {pick_device()}")

    print("\nbias baseline")
    base = BiasBaseline(rating_scale=scale).fit(train)
    results = {"baseline": {"val_rmse": rmse(val.rating, base.predict(val)),
                            **evaluate_ranking(base.recommend(train, val.userId.unique(), max(args.k)),
                                               val, args.k)}}
    print("  ", results["baseline"])

    print("\nmatrix factorization")
    mf = MF(args.factors, args.reg, args.lr, args.batch_size, args.epochs, rating_scale=scale)
    mf.fit(train, val)
    results["mf"] = {"val_rmse": rmse(val.rating, mf.predict(val)),
                     **evaluate_ranking(mf.recommend(train, val.userId.unique(), max(args.k)),
                                        val, args.k)}
    print("  ", results["mf"])

    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"mf_{args.dataset}_{args.split}_f{args.factors}"
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump({"settings": vars(args), "results": results, "history": mf.history}, f, indent=2)
    mf.user_embeddings().to_parquet(RESULTS_DIR / f"{name}_users.parquet", index=False)
    mf.item_embeddings().to_parquet(RESULTS_DIR / f"{name}_items.parquet", index=False)
    print(f"\nsaved results/{name}.json and embeddings")