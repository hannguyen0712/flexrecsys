import numpy as np


def bocpd(x, hazard=1/50, mu0=0.0, var0=4.0, obs_var=1.0):
    T = len(x)
    R = np.zeros((T + 1, T + 1))  # R[r,t] = P(run_length=r | x_{0:t-1})
    R[0, 0] = 1.0
    mu = np.array([mu0])   # posterior mean for each run length
    var = np.array([var0]) # posterior variance for each run length
    cp_prob = np.zeros(T)
    map_rl = np.zeros(T, dtype=int)

    for t in range(T):
        # predictive likelihood under each run length
        pred_var = var + obs_var
        pred = np.exp(-0.5 * (x[t] - mu) ** 2 / pred_var) / np.sqrt(2 * np.pi * pred_var)

        # growth: run continues | changepoint: run resets
        growth = R[:t + 1, t] * pred * (1 - hazard)
        cp = np.sum(R[:t + 1, t] * pred * hazard)
        R[1:t + 2, t + 1] = growth
        R[0, t + 1] = cp

        R[:, t + 1] /= R[:, t + 1].sum()  # normalize

        # Bayesian update of Normal posterior
        new_var = 1.0 / (1.0 / var + 1.0 / obs_var)
        new_mu = new_var * (mu / var + x[t] / obs_var)
        mu = np.concatenate([[mu0], new_mu])
        var = np.concatenate([[var0], new_var])

        cp_prob[t] = R[0, t + 1]        # P(changepoint at t)
        map_rl[t] = R[:, t + 1].argmax() # most likely run length

    return R, cp_prob, map_rl


def compute_residuals(model, train_df):
    # Compute prediction errors: residual = actual - predicted.
    predictions = model.predict(train_df)
    residuals = train_df["rating"].values - predictions
    return train_df[["userId", "timestamp", "movieId"]].assign(residual=residuals)


def detect_user_changepoints(residuals_df, hazard=1/50):
    # Detect changepoints per user where MAP run length drops > 5.
    changepoints = {}
    for user, group in residuals_df.groupby('userId'):
        group = group.sort_values(['timestamp', 'movieId'])
        _, _, map_rl = bocpd(group.residual.values, hazard=hazard)
        drops = [t for t in range(1, len(map_rl)) if map_rl[t] < map_rl[t-1] - 5]
        changepoints[user] = drops
    return changepoints


def generate_weights(train_df, changepoints, base_weight=0.5, post_cp_weight=1.0):
    weights = np.full(len(train_df), post_cp_weight)
    for user, cp_times in changepoints.items():
        if cp_times:
            user_idx = train_df[train_df.userId == user].sort_values(['timestamp', 'movieId']).index
            weights[user_idx[:cp_times[-1]]] = base_weight
    return weights


def compute_user_blending_weights(residuals_df, hazard=1/50, method='expected_rl'):
    # compute blending weights from run length posterior for embedding blending.
    alphas = {}
    changepoint_indices = {}

    for user, group in residuals_df.groupby('userId'):
        group = group.sort_values(['timestamp', 'movieId'])
        x = group.residual.values
        T = len(x)
        if T < 2:
            alphas[user] = 1.0
            changepoint_indices[user] = 0
            continue

        R, _, map_rl = bocpd(x, hazard=hazard)
        final_posterior = R[:T+1, T]  # P(r | all data)

        # α = E[run_length]/T: high α → stable, low α → recent shift
        if method == 'expected_rl':
            expected_rl = np.sum(np.arange(T+1) * final_posterior)
            alpha = expected_rl / T
        elif method == 'prob_long':
            alpha = final_posterior[T // 2:].sum()
        else:
            alpha = map_rl[-1] / T

        alphas[user] = np.clip(alpha, 0.0, 1.0)
        drops = [t for t in range(1, len(map_rl)) if map_rl[t] < map_rl[t-1] - 5]
        changepoint_indices[user] = drops[-1] if drops else 0

    return alphas, changepoint_indices