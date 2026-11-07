
import numpy as np


def bocpd(x, hazard=1 / 50, mu0=0.0, var0=4.0, obs_var=1.0):
    T = len(x)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    mu = np.array([mu0])
    var = np.array([var0])
    cp_prob = np.zeros(T)
    map_rl = np.zeros(T, dtype=int)

    for t in range(T):
        # 1. predictive probability of x_t under each run length (Normal)
        pred_var = var + obs_var
        pred = np.exp(-0.5 * (x[t] - mu) ** 2 / pred_var) / np.sqrt(2 * np.pi * pred_var)

        # 2. growth: run continues. changepoint: run resets to 0
        growth = R[: t + 1, t] * pred * (1 - hazard)
        cp = np.sum(R[: t + 1, t] * pred * hazard)
        R[1 : t + 2, t + 1] = growth
        R[0, t + 1] = cp

        # 3. normalise
        R[:, t + 1] /= R[:, t + 1].sum()

        # 4. update Normal posterior for each run length
        new_var = 1.0 / (1.0 / var + 1.0 / obs_var)
        new_mu = new_var * (mu / var + x[t] / obs_var)
        mu = np.concatenate([[mu0], new_mu])
        var = np.concatenate([[var0], new_var])

        cp_prob[t] = R[0, t + 1]
        map_rl[t] = R[:, t + 1].argmax()

    return R, cp_prob, map_rl


# compute r - r_hat for each rating, grouped by user.
def compute_residuals(model, train_df):
    predictions = model.predict(train_df)
    residuals = train_df["rating"].values - predictions
    return train_df[["userId", "timestamp"]].assign(residual=residuals)


# run bocpd on the residuals for each user
def detect_user_changepoints(residuals_df, hazard=1/50):
    changepoints = {}
    for user, group in residuals_df.groupby('userId'):
        group = group.sort_values('timestamp')
        _, cp_prob, map_rl = bocpd(group.residual.values, hazard=hazard)
        # Detect drops in MAP run length
        drops = [t for t in range(1, len(map_rl)) 
                 if map_rl[t] < map_rl[t-1] - 5]
        changepoints[user] = drops
    return changepoints

# assign weight to ratings
def generate_weights(train_df, changepoints, base_weight=0.5, post_cp_weight=1.0):
    weights = np.full(len(train_df), base_weight)
    for user, cp_times in changepoints.items():
        user_mask = train_df.userId == user
        user_idx = train_df[user_mask].sort_values('timestamp').index
        if cp_times:
            last_cp = cp_times[-1]  
            weights[user_idx[last_cp:]] = post_cp_weight
    return weights