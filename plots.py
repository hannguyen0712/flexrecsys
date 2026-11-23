# python plots.py results/compare_movielens_dev.json

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OURS_COLOR = "#2a78d6"
# display names, shared with compare.py; they override the names stored in older results files
LABELS = {"bias": "Bias baseline", "mf": "MF", "mf-refit": "flexrecsys (mf-refit only)",
          "mf-decay": "MF + time decay", "decay-refit": "Time decay, users refit",
          "timesvdpp": "TimeSVD++", "flexrecsys-v1": "flexrecsys (window BOCPD)",
          "flexrecsys-soft": "flexrecsys (soft update)",
          "flexrecsys-level": "flexrecsys (level update)", "flexrecsys": "flexrecsys (hard update)"}


def is_flex(model):
    return model == "mf-refit" or model.startswith("flexrecsys")


def model_labels(res):
    return {**res.get("labels", {}), **{k: v for k, v in LABELS.items() if k in res["runs"]}}


def model_order(res):
    # comparison methods first, then the flexrecsys models, with the reference model last
    flex = [m for m in res["runs"] if is_flex(m)]
    flex = [m for m in flex if m != "mf-refit"] + [m for m in flex if m == "mf-refit"]
    return [m for m in res["runs"] if not is_flex(m)] + flex
BASE_COLOR = "#52514e"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BASE_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

LOWER_IS_BETTER = {"val_rmse"}


def metric_label(m):
    if m == "val_rmse":
        return "RMSE"
    name, k = m.split("@")
    return f"{'Recall' if name == 'recall' else 'NDCG'}@{k}"


def metric_list(res):
    ks = res.get("ks", [10, 20])
    return ["val_rmse"] + [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]


def load_results(path):
    path = Path(path)
    res = json.loads(path.read_text())
    pu = path.with_name(path.stem + "_per_user.csv")
    return res, (pd.read_csv(pu) if pu.exists() else None)


def _bootstrap_mean(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def summarize(res, per_user=None, n_boot=2000):
    # long frame: model, metric, mean, lo, hi, interval
    rows = []
    for model, runs in res["runs"].items():
        df = pd.DataFrame(runs)
        for m in metric_list(res):
            if m not in df:
                continue
            vals = df[m].to_numpy(float)
            mean, lo, hi, kind = vals.mean(), vals.mean(), vals.mean(), "none"
            if len(vals) >= 2:
                sd = vals.std(ddof=1)
                lo, hi, kind = mean - sd, mean + sd, f"±1 sd over {len(vals)} seeds"
            elif per_user is not None and m != "val_rmse":
                x = per_user.loc[per_user.model == model, m].to_numpy(float)
                if len(x):
                    lo, hi = _bootstrap_mean(x, n_boot)
                    kind = "95% bootstrap CI over users"
            rows.append({"model": model, "metric": m, "mean": mean, "lo": lo, "hi": hi,
                         "interval": kind})
    return pd.DataFrame(rows)


def best_baseline(summary, ours, metric):
    s = summary[(summary.metric == metric) & (summary.model != ours)]
    if s.empty:
        return None
    return s.loc[s["mean"].idxmin() if metric in LOWER_IS_BETTER else s["mean"].idxmax(), "model"]


def deltas(res, per_user=None, summary=None, n_boot=2000):
    # ours vs best baseline per metric, relative change in %, paired bootstrap CI over users
    ours = res.get("ours")
    if not ours or ours not in res["runs"]:
        return pd.DataFrame()
    summary = summarize(res, per_user) if summary is None else summary
    rows = []
    rng = np.random.default_rng(0)
    for m in metric_list(res):
        base = best_baseline(summary, ours, m)
        if base is None:
            continue
        mo = summary.query("model == @ours and metric == @m")["mean"].iloc[0]
        mb = summary.query("model == @base and metric == @m")["mean"].iloc[0]
        row = {"metric": m, "baseline": base, "ours": mo, "base": mb,
               "rel": (mo - mb) / mb * 100, "lo": np.nan, "hi": np.nan}
        if per_user is not None and m != "val_rmse":
            # average each user over seeds first, then pair users present in both
            pu = per_user[per_user.model.isin([ours, base])]
            wide = pu.groupby(["userId", "model"])[m].mean().unstack().dropna()
            a, b = wide[ours].to_numpy(), wide[base].to_numpy()
            idx = rng.integers(0, len(a), (n_boot, len(a)))
            rel = (a[idx].mean(1) - b[idx].mean(1)) / b[idx].mean(1) * 100
            row["lo"], row["hi"] = np.percentile(rel, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(mean, lo, hi, digits):
    if np.isclose(lo, hi):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {(hi - lo) / 2:.{digits}f}"


def comparison_table(res, per_user=None, fmt="markdown", digits=4):
    # fmt: "df" (numbers), "markdown", "latex" or "csv"; best value per column is bolded
    summary = summarize(res, per_user)
    labels = model_labels(res)
    metrics = [m for m in metric_list(res) if m in set(summary.metric)]
    order = model_order(res)

    if fmt == "df":
        wide = summary.pivot(index="model", columns="metric", values="mean")
        return wide.loc[order, metrics].rename(index=labels, columns=metric_label)

    cells = {}
    for m in metrics:
        s = summary[summary.metric == m].set_index("model")
        best = s["mean"].idxmin() if m in LOWER_IS_BETTER else s["mean"].idxmax()
        for model in order:
            if model not in s.index:
                cells[(model, m)] = "–"
                continue
            r = s.loc[model]
            txt = _fmt(r["mean"], r["lo"], r["hi"], digits)
            if model == best and fmt != "csv":
                txt = f"**{txt}**" if fmt == "markdown" else f"\\textbf{{{txt}}}"
            cells[(model, m)] = txt

    header = ["Model"] + [metric_label(m) + (" ↓" if m in LOWER_IS_BETTER else " ↑")
                          for m in metrics]
    body = [[labels.get(model, model)] + [cells[(model, m)] for m in metrics] for model in order]

    d = deltas(res, per_user, summary)
    if not d.empty:
        d = d.set_index("metric")
        row = [f"Δ {labels.get(res['ours'], res['ours'])} vs best other"]
        for m in metrics:
            if m not in d.index:
                row.append("–")
                continue
            r = d.loc[m]
            txt = f"{r.rel:+.1f}%"
            if not np.isnan(r.lo):
                txt += f" [{r.lo:+.1f}, {r.hi:+.1f}]"
            row.append(txt)
        body.append(row)

    if fmt == "csv":
        return pd.DataFrame(body, columns=header).to_csv(index=False)
    if fmt == "latex":
        esc = lambda t: t.replace("%", r"\%").replace("±", r"$\pm$").replace("Δ", r"$\Delta$") \
            .replace("↓", r"$\downarrow$").replace("↑", r"$\uparrow$")
        lines = [r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
                 " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
        for i, row in enumerate(body):
            if not d.empty and i == len(body) - 1:
                lines.append(r"\midrule")
            lines.append(" & ".join(esc(c) for c in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + ["---:"] * len(metrics)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_comparison(res, per_user=None, metrics=None, out=None, title=None, max_cols=5):
    # dot plot, one panel per metric, models as rows; dots don't need a zero baseline,
    # so small gaps between models stay visible. flexrecsys models are the coloured rows.
    summary = summarize(res, per_user)
    metrics = metrics or [m for m in metric_list(res) if m in set(summary.metric)]
    labels = model_labels(res)
    order = model_order(res)  # flexrecsys models on the top rows
    y = {m: i for i, m in enumerate(order)}

    ncols = min(len(metrics), max_cols)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.2 + 2.3 * ncols, 0.5 + nrows * (0.75 + 0.42 * len(order))),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, metrics):
        _style(ax)
        s = summary[summary.metric == m].set_index("model")
        best = s["mean"].idxmin() if m in LOWER_IS_BETTER else s["mean"].idxmax()
        for model in order:
            if model not in s.index:
                continue
            r = s.loc[model]
            is_ours = is_flex(model)
            c = OURS_COLOR if is_ours else BASE_COLOR
            ax.plot([r.lo, r.hi], [y[model]] * 2, color=c, linewidth=2 if is_ours else 1.4,
                    solid_capstyle="round", alpha=0.9 if is_ours else 0.55, zorder=2)
            ax.scatter(r["mean"], y[model], s=64 if is_ours else 40, color=c, zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
            if model == best:
                ax.annotate(f"{r['mean']:.4f}", (r["mean"], y[model]), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.5, va="bottom",
                            color=INK, fontweight="bold")
        lo, hi = s["lo"].min(), s["hi"].max()
        pad = (hi - lo) * 0.15 or abs(hi) * 0.02 or 0.01
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_title(f"{metric_label(m)}  {'lower' if m in LOWER_IS_BETTER else 'higher'} is better",
                     fontsize=9, color=INK, loc="left", pad=10)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels.get(m, m) for m in order], fontsize=9)
        ax.tick_params(labelleft=True)
        for t, m in zip(ax.get_yticklabels(), order):
            t.set_color(OURS_COLOR if is_flex(m) else INK_2)
            t.set_fontweight("bold" if is_flex(m) else "normal")
        ax.set_ylim(-0.7, len(order) - 0.4)

    kinds = sorted(set(summary.interval) - {"none"})
    split = "test" if res.get("eval") == "test" else "validation"
    d = res.get("data", {})
    note = (f"{res.get('dataset', '')}: {d.get('train_ratings', 0):,} train / "
            f"{d.get('heldout_ratings', d.get('val_ratings', 0)):,} {split} ratings, "
            f"{d.get('users', 0):,} users. "
            f"Whiskers: {'; '.join(kinds) if kinds else 'none (single run)'}. Scored on {split}.")
    fig.text(0.01, 0.01, note, fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.suptitle(title or f"{split.capitalize()} metrics: flexrecsys models (blue) and comparison methods", fontsize=11,
                 color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.05 / nrows, 1, 1 - 0.07 / nrows), w_pad=1.6, h_pad=1.2)
    if out:
        for o in np.atleast_1d(out):
            fig.savefig(o, dpi=200, facecolor=SURFACE)
        plt.close(fig)
    return fig


def plot_at_k(res, per_user=None, out=None):
    # recall and NDCG against K, one line per model; only useful with 3+ cutoffs.
    # lines usually bunch together at the right end, so identity comes from the legend + markers
    summary = summarize(res, per_user)
    ks = res.get("ks", [])
    labels = model_labels(res)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax, fam in zip(axes, ["recall", "ndcg"]):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        bi = 0
        for model in model_order(res):
            s = summary[(summary.model == model) & summary.metric.str.startswith(fam + "@")]
            if s.empty:
                continue
            s = s.assign(k=s.metric.str.split("@").str[1].astype(int)).sort_values("k")
            is_ours = is_flex(model)
            c = OURS_COLOR if is_ours else BASE_COLOR
            mk = BASE_MARKERS[bi % len(BASE_MARKERS)]
            bi += 1
            ax.fill_between(s.k, s.lo, s.hi, color=c, alpha=0.12 if is_ours else 0.06, linewidth=0)
            ax.plot(s.k, s["mean"], color=c, linewidth=2, marker=mk, markersize=6 if is_ours else 5,
                    alpha=1 if is_ours else 0.7, label=labels.get(model, model))
        ax.set_xticks(ks)
        ax.set_xlabel("K", fontsize=8, color=MUTED)
        ax.set_title(f"{'Recall' if fam == 'recall' else 'NDCG'}@K", fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    if out:
        for o in np.atleast_1d(out):
            fig.savefig(o, dpi=200, facecolor=SURFACE)
        plt.close(fig)
    return fig


def plot_shift_curve(curve, refs, models, out):
    # left: gain over mf-refit against the share of synthetic users shifted.
    # right: the same gains against the excess flagged share (flagged minus flagged after
    # shuffling), which is measured the same way on real data, with MovieLens and Netflix added
    colors = [OURS_COLOR, BASE_COLOR, MUTED]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, xcol, xlabel in ((axes[0], "shift_frac", "synthetic users with a planted shift (%)"),
                             (axes[1], "excess_flagged", "users flagged beyond the shuffled rate (%)")):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.axhline(0, color=AXIS, linewidth=1)
        c = curve.sort_values(xcol)
        for m, color, marker in zip(models, colors, BASE_MARKERS):
            x = 100 * c[xcol]
            ax.errorbar(x, c[m], yerr=[c[m] - c[f"{m}_lo"], c[f"{m}_hi"] - c[m]], color=color,
                        marker=marker, markersize=5, linewidth=1.5, capsize=3, label=m)
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    axes[0].set_ylabel("Recall@10 change over mf-refit (%)", color=INK_2, fontsize=9)
    for _, r in refs.iterrows():
        x, y = 100 * r.excess_flagged, r.flexrecsys
        axes[1].errorbar([x], [y], yerr=[[y - r.flexrecsys_lo], [r.flexrecsys_hi - y]],
                         color=INK, marker="*", markersize=11, capsize=3, linestyle="none")
        axes[1].annotate(f"{r.dataset}, flexrecsys", (x, y), textcoords="offset points",
                         xytext=(-8, 8), ha="right", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("When does change-point weighting pay off?", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    paths = [Path(f"{out}.png"), Path(f"{out}.pdf")]
    for p in paths:
        fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    return paths


def render_all(json_path):
    json_path = Path(json_path)
    res, per_user = load_results(json_path)
    stem = json_path.with_suffix("")
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    plot_comparison(res, per_user, out=paths)
    if len(res.get("ks", [])) >= 3:
        p = Path(f"{stem}_at_k.png")
        plot_at_k(res, per_user, out=p)
        paths.append(p)
    for fmt, ext in [("markdown", "md"), ("latex", "tex"), ("csv", "csv")]:
        p = Path(f"{stem}_table.{ext}")
        p.write_text(comparison_table(res, per_user, fmt=fmt) + "\n")
        paths.append(p)
    return paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python plots.py results/compare_<name>.json [...]")
    for path in sys.argv[1:]:
        for p in render_all(path):
            print("wrote", p)