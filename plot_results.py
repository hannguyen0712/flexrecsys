"""Make MF vs. baseline figures from the output of tune_mf.py
    1_mf_vs_baseline.png   RMSE and ranking: baseline vs default MF vs best-RMSE MF vs best-ranking MF
    2_learning_curve.png   train/val RMSE per epoch for the best-RMSE MF
    3_tuning_heatmaps.png  val RMSE and Recall@K over reg x factors
    4_metrics_vs_k.png     Recall@K and NDCG@K against K
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "results"
DEFAULT = dict(factors=64, reg=0.05, lr=5e-3)   # mf.py's defaults
COLORS = {"Bias baseline": "#9aa0a6", "MF (default)": "#f4a261",
          "MF (best RMSE)": "#2a9d8f", "MF (best ranking)": "#264653"}


def pick(mf, **kw):
    m = mf
    for k, v in kw.items():
        m = m[np.isclose(m[k].astype(float), v)]
    return m.iloc[0] if len(m) else None


def desc(r):
    return f"f={int(r.factors)}, reg={r.reg:g}, lr={r.lr:g}"


def bar_panel(ax, entries, metrics, title, truncate=False):
    width = 0.8 / len(entries)
    x = np.arange(len(metrics))
    allv = []
    for j, (name, row) in enumerate(entries):
        vals = [row[m] for m in metrics]
        allv += vals
        bars = ax.bar(x + j * width, vals, width, label=name, color=COLORS[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(entries) - 1) / 2, metrics)
    if truncate:
        ax.set_ylim(max(0, min(allv) - 0.08), max(allv) + 0.02)
        title += "  (axis does not start at 0)"
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movielens")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", default=None, help="same tag tune_mf.py used (default: dataset_split)")
    args = ap.parse_args()
    tag = args.tag or f"{args.dataset}_{args.split}"

    tdir = ROOT / "tuning"
    df = pd.read_csv(tdir / f"tuning_{tag}.csv")
    histories = json.loads((tdir / f"tuning_{tag}_history.json").read_text())
    out = ROOT / "figures" / tag
    out.mkdir(parents=True, exist_ok=True)

    ks = sorted(int(c.split("@")[1]) for c in df.columns if c.startswith("recall@"))
    k0 = 10 if 10 in ks else ks[0]
    base = df[df.model == "baseline"].iloc[0]
    mf = df[df.model == "mf"]
    best = mf.sort_values("val_rmse").iloc[0]
    best_rank = mf.sort_values(f"recall@{k0}", ascending=False).iloc[0]
    default = pick(mf, **DEFAULT)
    print("best val RMSE   :", desc(best), f"-> RMSE {best.val_rmse:.4f}, recall@{k0} {best[f'recall@{k0}']:.4f}")
    print("best recall@%d  :" % k0, desc(best_rank), f"-> RMSE {best_rank.val_rmse:.4f}, recall@{k0} {best_rank[f'recall@{k0}']:.4f}")

    entries = [("Bias baseline", base)]
    if default is not None:
        entries.append(("MF (default)", default))
    entries.append(("MF (best RMSE)", best))
    if desc(best_rank) != desc(best):
        entries.append(("MF (best ranking)", best_rank))

    # baseline vs MF
    rank_metrics = [f"{m}@{k}" for k in ks[:2] for m in ("recall", "ndcg")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 3]})
    bar_panel(a1, entries, ["val_rmse"], "Val RMSE (lower is better)", truncate=True)
    bar_panel(a2, entries, rank_metrics, "Ranking on val (higher is better)")
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Bias baseline vs MF  |  {tag}  |  best RMSE: {desc(best)}  |  best ranking: {desc(best_rank)}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "1_mf_vs_baseline.png", dpi=160)

    # learning curve (best RMSE)
    h = pd.DataFrame(histories[f"f{int(best.factors)}_reg{best.reg}_lr{best.lr}"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(h.epoch, h.train_rmse, marker="o", label="train RMSE", color="#e76f51")
    ax.plot(h.epoch, h.val_rmse, marker="o", label="val RMSE", color="#2a9d8f")
    ax.axvline(best.best_epoch, ls="--", color="gray", lw=1)
    ax.text(best.best_epoch, ax.get_ylim()[1], f" best epoch {int(best.best_epoch)}",
            va="top", fontsize=8, color="gray")
    ax.axhline(base.val_rmse, ls=":", color="#9aa0a6")
    ax.text(h.epoch.max(), base.val_rmse, "bias baseline val RMSE ", ha="right", va="bottom",
            fontsize=8, color="#6b7075")
    ax.set(xlabel="epoch", ylabel="RMSE", title=f"Learning curve ({desc(best)})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "2_learning_curve.png", dpi=160)

    # tuning heatmaps (at the best-RMSE lr)
    sub = mf[np.isclose(mf.lr.astype(float), best.lr)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, cmap in [(axes[0], "val_rmse", "viridis_r"), (axes[1], f"recall@{k0}", "viridis")]:
        pv = sub.pivot_table(index="reg", columns="factors", values=metric)
        im = ax.imshow(pv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(pv.shape[1]), [int(c) for c in pv.columns])
        ax.set_yticks(range(pv.shape[0]), [f"{r:g}" for r in pv.index])
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)
        ax.set(xlabel="factors", ylabel="reg", title=f"{metric} (lr={best.lr:g})")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "3_tuning_heatmaps.png", dpi=160)

    # metrics vs K
    if len(ks) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, m in zip(axes, ("recall", "ndcg")):
            for name, row in entries:
                ax.plot(ks, [row[f"{m}@{k}"] for k in ks], marker="o", label=name, color=COLORS[name])
            ax.set(xlabel="K", title=f"{m.upper() if m == 'ndcg' else m.capitalize()}@K")
            ax.set_xticks(ks)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "4_metrics_vs_k.png", dpi=160)

    print(f"saved figures to {out}")


if __name__ == "__main__":
    main()