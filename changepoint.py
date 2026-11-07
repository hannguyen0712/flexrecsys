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


def time_decay_weights(train, half_life_days=180.0, floor=0.0):
    # continuous exponential decay from each user's latest train rating (the smooth-drift view)
    last = train.groupby("userId")["timestamp"].transform("max").to_numpy()
    age = (last - train.timestamp.to_numpy()) / 86_400
    return np.maximum(0.5 ** (age / half_life_days), floor)
