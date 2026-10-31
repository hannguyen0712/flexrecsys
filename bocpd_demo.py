"""
Implementation from Minimal Bayesian Online Change Point Detection (Adams & MacKay, 2007).
Demo: a simulated stream of one user's rating residuals whose mean shifts at t = 100.
"""
import numpy as np


def bocpd(x, hazard=1 / 50, mu0=0.0, var0=4.0, obs_var=1.0):
    T = len(x)
    # R[r, t] = P(run length = r | x_1..t)
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

# Demo usage of the bocpd function
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.8, 1.0, 100), rng.normal(-0.7, 1.0, 100)])

    R, cp_prob, map_rl = bocpd(x, hazard=1 / 100, obs_var=1.0)

    # A drop in the MAP run length is a stronger signal than a raw P(r_t = 0) spike
    drops = [t for t in range(1, len(x)) if map_rl[t] < map_rl[t - 1] - 5]
    print("True change point: t = 100")
    print("MAP run length drops at t =", drops)
    print("Max P(r_t = 0) after t=100: %.3f at t=%d" % (
        cp_prob[100:].max(), 100 + cp_prob[100:].argmax()))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        ax[0].plot(x, lw=1)
        ax[0].axvline(100, color="red", ls="--")
        ax[0].set_title("Simulated residual stream (true change at t=100)")
        ax[1].imshow(np.log(R[:120, :] + 1e-12), origin="lower", aspect="auto", cmap="gray_r")
        ax[1].set_ylabel("run length")
        ax[1].set_title("Run length posterior (log scale)")
        ax[2].plot(cp_prob)
        ax[2].set_ylabel("P(r_t = 0)")
        ax[2].set_xlabel("time")
        plt.tight_layout()
        plt.savefig("./bocpd_demo.png", dpi=120)
        print("saved bocpd_demo.png")
    except ImportError:
        pass