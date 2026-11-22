import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


def bocpd(X, hazard=1 / 20, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    # X: (T, D) sequence, dims treated as independent Normal-Gamma with Student-t predictive.
    # returns the run-length posterior after the last step: p[r-1] = P(last r points share a regime)
    X = np.atleast_2d(np.asarray(X, float))
    T, D = X.shape
    log_r = np.zeros(1)
    mu = np.full((1, D), mu0)
    kappa = np.full(1, kappa0)
    alpha = np.full(1, alpha0)
    beta = np.full((1, D), beta0)
    log_h, log_1h = np.log(hazard), np.log1p(-hazard)
    for t in range(T):
        x = X[t]
        # prior over run length before seeing x: index 0 = change now, k = continue a k-long run
        if t == 0:
            log_prior = np.zeros(1)
        else:
            log_prior = np.r_[logsumexp(log_r) + log_h, log_r + log_1h]
            mu = np.vstack([np.full((1, D), mu0), mu])
            kappa = np.r_[kappa0, kappa]
            alpha = np.r_[alpha0, alpha]
            beta = np.vstack([np.full((1, D), beta0), beta])
        nu = 2 * alpha
        scale2 = beta * (kappa + 1)[:, None] / (alpha * kappa)[:, None]
        z2 = (x - mu) ** 2 / scale2
        log_pred = (gammaln((nu + 1) / 2) - gammaln(nu / 2))[:, None] \
            - 0.5 * np.log(nu[:, None] * np.pi * scale2) \
            - ((nu + 1) / 2)[:, None] * np.log1p(z2 / nu[:, None])
        log_r = log_prior + log_pred.sum(1)
        log_r -= logsumexp(log_r)
        # conjugate update with x for every run length
        beta = beta + kappa[:, None] * (x - mu) ** 2 / (2 * (kappa + 1)[:, None])
        mu = (kappa[:, None] * mu + x) / (kappa + 1)[:, None]
        kappa = kappa + 1
        alpha = alpha + 0.5
    return np.exp(log_r)


def assign_windows(train, window=10):
    # window index per row within each user's time-ordered history; a short tail joins the last window
    order = train.sort_values(["userId", "timestamp", "movieId"]).index
    df = train.loc[order, ["userId"]]
    pos = df.groupby("userId").cumcount().to_numpy()
    n = df.groupby("userId")["userId"].transform("size").to_numpy()
    n_full = np.maximum(n // window, 1)
    tail = n - (n // window) * window
    last = np.where(tail >= window / 2, n_full, n_full - 1)
    win = np.minimum(pos // window, last)
    return pd.Series(win, index=order).reindex(train.index).to_numpy()


def window_signals(model, train, window=10, n_components=8, reg=5.0):
    # per (user, window): ridge user vector with items frozen, projected onto the top principal
    # directions of the item factors so the signal is low-dimensional and comparable across windows
    win = assign_windows(train, window)
    keys, uniq = pd.factorize(pd.MultiIndex.from_arrays([train.userId.to_numpy(), win]))
    frame = train.assign(userId=keys)
    vecs = model.user_vectors_from(frame, reg=reg).set_index("userId").sort_index()

    items = model.item_embeddings()
    Q = items.filter(regex=r"^q\d+$").to_numpy()
    _, _, Vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    P = vecs.filter(regex=r"^p\d+$").to_numpy()
    Z = np.hstack([P @ Vt[:n_components].T, vecs[["b_u"]].to_numpy()])

    keys = vecs.index.to_numpy()
    sig = pd.DataFrame(Z, columns=[f"z{j}" for j in range(Z.shape[1])])
    sig.insert(0, "window", uniq.get_level_values(1)[keys])
    sig.insert(0, "userId", uniq.get_level_values(0)[keys])
    return sig.sort_values(["userId", "window"]).reset_index(drop=True), win


def robust_standardize(X):
    # centre per dim, scale by within-regime noise estimated from first differences, so one
    # big jump between regimes doesn't inflate the scale the way a plain std would
    X = X - np.median(X, 0)
    if len(X) < 3:
        return X
    d = np.abs(np.diff(X, axis=0))
    sigma = np.median(d, 0) / (0.6745 * np.sqrt(2))
    return X / np.maximum(sigma, 1e-3)


def changepoint_weights(model, train, window=10, hazard=1 / 20, floor=0.2, n_components=8,
                        reg=5.0, return_diagnostics=False):
    # model: a fitted MF (needs user_vectors_from and item_embeddings)
    sig, win = window_signals(model, train, window, n_components, reg)
    feat = [c for c in sig.columns if c not in ("userId", "window")]
    w_by = {}
    diag = []
    for user, g in sig.groupby("userId", sort=False):
        X = g[feat].to_numpy()
        T = len(X)
        if T < 3:
            w_by[user] = np.ones(T)
            diag.append((user, T, T, 0.0))
            continue
        p = bocpd(robust_standardize(X), hazard=hazard)
        # window j (0-based) is in the current regime iff the final run length >= T - j
        p_in = np.array([p[T - j - 1:].sum() for j in range(T)])
        w_by[user] = floor + (1 - floor) * p_in
        diag.append((user, T, int(np.argmax(p)) + 1, float(1 - p[-1])))

    users = train.userId.to_numpy()
    weights = np.fromiter((w_by[u][k] for u, k in zip(users, win)), float, len(train))
    if return_diagnostics:
        d = pd.DataFrame(diag, columns=["userId", "n_windows", "map_run_length", "p_change"])
        return weights, d
    return weights


def segment_evidence(S, b, yy, n, lam, sigma2):
    # log marginal likelihood of a segment under y ~ N(X theta, sigma2), theta ~ N(0, sigma2 / lam).
    # S = X'X, b = X'y, yy = y'y, batched over the leading axis
    d = S.shape[-1]
    A = S + lam * np.eye(d)
    sol = np.linalg.solve(A, b[..., None])[..., 0]
    logdet = np.linalg.slogdet(A)[1] - d * np.log(lam)
    return -0.5 * (n * np.log(2 * np.pi * sigma2) + logdet + (yy - (b * sol).sum(-1)) / sigma2)


def regime_features(model, n_components):
    # item factors projected on their top principal directions, plus a constant for the user bias
    items = model.item_embeddings()
    Q = items.filter(regex=r"^q\d+$").to_numpy()
    _, _, Vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    return np.hstack([Q @ Vt[:n_components].T, np.ones((len(Q), 1))]), items["b_i"].to_numpy()


def fit_regime_prior(model, train, n_components=8, grid=None):
    # empirical Bayes: the prior strength and rating noise that best explain the training ratings
    # with one vector per user. keeps the detector matched to the scale of the item factors,
    # which changes with how the base model was trained. no validation data involved
    grid = np.logspace(-3, 3, 25) if grid is None else grid
    Z, b_i = regime_features(model, n_components)
    i = model.items.encode(train.movieId)
    u = model.users.encode(train.userId)
    y = train.rating.to_numpy(float) - float(model.net.mu) - b_i[i]
    X = Z[i]
    d, n_users = X.shape[1], len(model.users)
    S, b, yy = np.zeros((n_users, d, d)), np.zeros((n_users, d)), np.zeros(n_users)
    np.add.at(S, u, X[:, :, None] * X[:, None, :])
    np.add.at(b, u, X * y[:, None])
    np.add.at(yy, u, y * y)
    n = np.bincount(u, minlength=n_users)
    best = None
    for lam in grid:
        sol = np.linalg.solve(S + lam * np.eye(d), b[..., None])[..., 0]
        # for a given lam the evidence-maximising noise variance has this closed form
        sigma2 = (yy - (b * sol).sum(-1)).sum() / n.sum()
        evidence = segment_evidence(S, b, yy, n, lam, sigma2).sum()
        if best is None or evidence > best[0]:
            best = (evidence, float(lam), float(np.sqrt(sigma2)))
    return best[1], best[2]


def regime_weights(model, train, prior=0.2, n_components=8, reg=None, min_seg=10, floor=0.0,
                   noise=None, return_diagnostics=False):
    # at most one change per user, located in rating space with items frozen: evidence for one
    # user vector over the whole history vs two vectors split at each candidate cut.
    # weight = posterior probability that the rating belongs to the user's current regime
    # reg=None fits the prior strength (and the noise, unless given) from the training ratings
    if reg is None:
        reg, fitted_noise = fit_regime_prior(model, train, n_components)
        noise = fitted_noise if noise is None else noise
    Z, b_i = regime_features(model, n_components)

    df = train.sort_values(["userId", "timestamp", "movieId"])
    idx = model.items.encode(df.movieId)
    y_all = df.rating.to_numpy(float) - float(model.net.mu) - b_i[idx]
    if noise is None:
        noise = np.sqrt(np.mean((df.rating.to_numpy(float) - model.predict(df)) ** 2))
    sigma2 = float(noise) ** 2
    X = Z[idx]
    users = df.userId.to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(users)) + 1, len(users)]

    w = np.ones(len(df))
    diag = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        n = hi - lo
        if n < 2 * min_seg:
            diag.append((users[lo], n, -1, 0.0))
            continue
        x, y = X[lo:hi], y_all[lo:hi]
        cS = np.cumsum(x[:, :, None] * x[:, None, :], 0)
        cb = np.cumsum(x * y[:, None], 0)
        cy = np.cumsum(y * y)
        cuts = np.arange(min_seg, n - min_seg + 1)  # cut c: ratings c.. are the new regime
        k = cuts - 1
        pre = segment_evidence(cS[k], cb[k], cy[k], cuts, reg, sigma2)
        post = segment_evidence(cS[-1] - cS[k], cb[-1] - cb[k], cy[-1] - cy[k], n - cuts, reg, sigma2)
        none = segment_evidence(cS[-1], cb[-1], cy[-1], n, reg, sigma2)
        logp = np.r_[np.log1p(-prior) + none, np.log(prior / len(cuts)) + pre + post]
        p = np.exp(logp - logp.max())
        p /= p.sum()
        p_cut = np.zeros(n)
        p_cut[cuts] = p[1:]
        w[lo:hi] = floor + (1 - floor) * (p[0] + np.cumsum(p_cut))
        diag.append((users[lo], n, int(cuts[np.argmax(p[1:])]), float(1 - p[0])))
    weights = pd.Series(w, index=df.index).reindex(train.index).to_numpy()
    if return_diagnostics:
        d = pd.DataFrame(diag, columns=["userId", "n_ratings", "map_cut", "p_change"])
        return weights, d
    return weights


def time_decay_weights(train, half_life_days=180.0, floor=0.0):
    # continuous exponential decay from each user's latest train rating (the smooth-drift view)
    last = train.groupby("userId")["timestamp"].transform("max").to_numpy()
    age = (last - train.timestamp.to_numpy()) / 86_400
    return np.maximum(0.5 ** (age / half_life_days), floor)

# shift diagnostics: whether the detector's flags are real, and what changes at a detected change?

def mean_interval(x, n_boot=4000, seed=0):
    # mean and 95% bootstrap interval
    x = np.asarray(x, float)
    means = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), *(float(v) for v in np.percentile(means, [2.5, 97.5]))


def shuffle_within_users(train, seed=0):
    # placebo: random rating order per user, so every change still flagged is a false alarm
    rng = np.random.default_rng(seed)
    return train.assign(timestamp=train.groupby("userId").timestamp.transform(
        lambda s: rng.permutation(s.to_numpy())))


def _window(train, diag, max_k, after):
    # the k ratings just before (or just after) each user's detected change
    df = train.sort_values(["userId", "timestamp", "movieId"])
    pos = df.groupby("userId").cumcount()
    d = diag[diag.map_cut >= 0].set_index("userId")
    cut = df.userId.map(d.map_cut)
    k = np.minimum(np.minimum(cut, df.userId.map(d.n_ratings) - cut), max_k)
    keep = (pos >= cut) & (pos < cut + k) if after else (pos >= cut - k) & (pos < cut)
    return df[keep & (k >= 5)]


def before_after_gap(model, train, val, diag, reg, max_k=50):
    items = model.item_embeddings()
    Q = items.filter(regex=r"^q\d+$").to_numpy()
    b_i = items["b_i"].to_numpy()
    idx = model.items.encode(val.movieId)
    v, idx = val[idx >= 0], idx[idx >= 0]

    def user_mse(ratings):
        vec = model.user_vectors_from(ratings, reg).set_index("userId")
        u = v.userId.to_numpy()
        pred = (float(model.net.mu) + b_i[idx] + vec.b_u.reindex(u).to_numpy()
                + (vec.filter(regex=r"^p\d+$").reindex(u).to_numpy() * Q[idx]).sum(1))
        return pd.Series((v.rating.to_numpy() - pred) ** 2).groupby(u).mean()

    return (user_mse(_window(train, diag, max_k, False))
            - user_mse(_window(train, diag, max_k, True))).dropna()


def genre_shift(train, movies, diag, max_k=50, n_perm=50, seed=0):
    # Jensen-Shannon divergence between the genre mix before and after the change, minus its average over random splits of the same ratings
    rng = np.random.default_rng(seed)
    names = sorted({g for gs in movies.genres for g in gs})
    col = {g: j for j, g in enumerate(names)}
    G = np.zeros((len(movies), len(names)))
    for r, gs in enumerate(movies.genres):
        G[r, [col[g] for g in gs]] = 1.0 / len(gs)
    table = pd.DataFrame(G, index=movies.movieId.to_numpy(), columns=names)

    def js(a, b):
        p, q = a.mean(0), b.mean(0)
        p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
        m = (p + q) / 2
        kl = lambda x: float((x[x > 0] * np.log2(x[x > 0] / m[x > 0])).sum())
        return (kl(p) + kl(q)) / 2

    before, after = _window(train, diag, max_k, False), _window(train, diag, max_k, True)
    rows, moves = [], []
    for user, a in after.groupby("userId"):
        b = before[before.userId == user]
        Gb = table.reindex(b.movieId).fillna(0.0).to_numpy()
        Ga = table.reindex(a.movieId).fillna(0.0).to_numpy()
        both, k = np.vstack([Gb, Ga]), len(Gb)
        null = [js(*np.split(both[rng.permutation(2 * k)], [k])) for _ in range(n_perm)]
        real = js(Gb, Ga)
        rows.append((user, real - np.mean(null), real > np.percentile(null, 95)))
        moves.append(Ga.mean(0) - Gb.mean(0))
    out = pd.DataFrame(rows, columns=["userId", "excess", "beyond_chance"]).set_index("userId")
    return out, pd.DataFrame(moves, index=out.index, columns=names)