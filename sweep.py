"""Run a named experiment (a grid of configurations x N runs) and save the results.

    python sweep.py list
    python sweep.py budget            # one experiment
    python sweep.py all --runs 10     # everything
    python sweep.py arch --workers 4

Outputs in ./results/: <exp>_summary.csv (mean/sd of loss + accuracy + precision/recall/F1
per config) and <exp>_hist.json (best training loss per iteration, for convergence plots).

The grids below are STARTING POINTS. Choosing the ranges, justifying them from the
literature and interpreting the results is your job (and must be your own work).
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from run import DEFAULTS, one_run, prepare_split

DATA = "data_banknote_authentication.txt"
H = lambda hidden, act: dict(layers=[4] + hidden + [1], acts=[act] * len(hidden) + ["logistic"])

EXPERIMENTS = {
    # Q1: architecture (layers, neurons, activation)
    "arch": {f"{h}-{a}": H(h, a) for h in ([1], [2], [4], [8], [4, 4], [8, 8])
             for a in ("logistic", "relu", "tanh")},
    # Q2: fixed budget of 500 evaluations split different ways
    "budget": {f"swarm{s}x{500 // s}it": dict(swarm=s, iters=500 // s)
               for s in (5, 10, 20, 25, 50, 100)},
    # Q3: cognitive (beta) vs informant (gamma) balance, and the global term (delta)
    "coef": {**{f"b{b}_g{g}": dict(beta=b, gamma=g)
                for b, g in ((0.5, 2.5), (1.0, 2.0), (1.5, 1.5), (2.0, 1.0), (2.5, 0.5))},
             **{f"delta{d}": dict(delta=d) for d in (0.0, 0.5, 1.0, 2.0)}},
    # extras
    "informants": {f"inf{k}": dict(informants=k) for k in (1, 2, 3, 5, 10)},
    "boundary": {b: dict(boundary=b) for b in ("none", "clip", "reflect")},
    "learned_acts": {"fixed-tanh": H([4], "tanh"),
                     "learned": dict(layers=[4, 4, 1], acts=["tanh", "logistic"], learn_acts=True),
                     "fixed-relu": H([4], "relu"), "fixed-logistic": H([4], "logistic")},
    # which training loss PSO minimises
    "loss": {m: dict(metric=m) for m in ("mse", "bce", "error")},
}

_split = None


def _init(split):
    global _split
    _split = split


def _job(args):
    name, cfg, seed = args
    return name, seed, one_run(cfg, seed, _split)


def run_experiment(exp, runs, workers, data):
    grid = EXPERIMENTS[exp]
    jobs = [(n, c, s) for n, c in grid.items() for s in range(runs)]
    split = prepare_split(data)       # load in the main process so a bad path fails fast
    with Pool(workers, initializer=_init, initargs=(split,)) as pool:
        out = pool.map(_job, jobs)
    os.makedirs("results", exist_ok=True)

    # metrics to average over the runs of each configuration
    keys = ["train", "test", "train_loss", "precision", "recall", "f1"]
    header = ["name", "config", "runs"]
    for k in keys:
        header += [f"{k}_mean", f"{k}_sd"]
    header.append("evals")

    hist, rows = {}, []
    for name in grid:
        rs = [r for n, s, r in sorted(out, key=lambda t: (t[0], t[1])) if n == name]
        hist[name] = [r["history"] for r in rs]                 # convergence (loss per iteration)
        stat = {k: np.array([r[k] for r in rs]) for k in keys}  # one array per metric
        row = [name, {**DEFAULTS, **grid[name]}, runs]
        for k in keys:
            row += [stat[k].mean(), stat[k].std(ddof=1)]
        row.append(rs[0]["evals"])
        rows.append(row)
        print(f"{exp:12s} {name:18s} loss {stat['train_loss'].mean():.4f}  "
              f"test {stat['test'].mean():.4f}±{stat['test'].std(ddof=1):.4f}  "
              f"F1 {stat['f1'].mean():.4f}")

    with open(f"results/{exp}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(f"results/{exp}_hist.json", "w") as f:
        json.dump(hist, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", help="list | all | " + " | ".join(EXPERIMENTS))
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--data", default=DATA)
    a = p.parse_args()
    if a.experiment == "list":
        for e, g in EXPERIMENTS.items():
            print(f"{e}: {len(g)} configs -> {list(g)}")
    else:
        for e in (EXPERIMENTS if a.experiment == "all" else [a.experiment]):
            run_experiment(e, a.runs, a.workers, a.data)
