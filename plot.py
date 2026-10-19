"""Make figures from sweep.py output.   python plot.py budget   (or: python plot.py all)

Writes results/<exp>_bars.png (mean test/train accuracy, error bars = 1 sd over runs)
and results/<exp>_convergence.png (mean best training fitness vs evaluations).
"""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(exp):
    rows = list(csv.DictReader(open(f"results/{exp}_summary.csv")))
    hist = json.load(open(f"results/{exp}_hist.json"))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names) + 2), 4))
    for off, key, lab in ((-0.2, "train", "train"), (0.2, "test", "test")):
        ax.bar(x + off, [float(r[f"{key}_mean"]) for r in rows], 0.4,
               yerr=[float(r[f"{key}_sd"]) for r in rows], capsize=2, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("accuracy"); ax.set_title(f"{exp}: mean ± sd over runs")
    lo = min(float(r["test_mean"]) for r in rows)
    ax.set_ylim(max(0.0, lo - 0.1), 1.005); ax.legend()
    fig.tight_layout(); fig.savefig(f"results/{exp}_bars.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in rows:
        h = np.array(hist[r["name"]])                         # runs x iterations
        swarm = json.loads(r["config"].replace("'", '"').replace("True", "true")
                           .replace("False", "false"))["swarm"]
        ev = (np.arange(h.shape[1]) + 1) * swarm
        m, s = h.mean(0), h.std(0, ddof=1)
        ax.plot(ev, m, label=r["name"]); ax.fill_between(ev, m - s, m + s, alpha=0.15)
    ax.set_xlabel("fitness evaluations"); ax.set_ylabel("best training accuracy")
    ax.set_title(f"{exp}: convergence"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"results/{exp}_convergence.png", dpi=150); plt.close(fig)
    print("wrote", f"results/{exp}_bars.png", f"results/{exp}_convergence.png")


if __name__ == "__main__":
    import glob
    exps = ([p.split("/")[-1][:-12] for p in glob.glob("results/*_summary.csv")]
            if sys.argv[1] == "all" else [sys.argv[1]])
    for e in exps:
        plot(e)
