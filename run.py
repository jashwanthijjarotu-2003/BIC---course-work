"""Task 3 + 4: couple PSO to the ANN and train/evaluate on the banknote data.

Coupling: each PSO particle is a flat vector of length ann.n_params. To evaluate a
particle we ann.set_params(particle) and measure the ANN's accuracy on the TRAINING set;
that accuracy is the particle's fitness. The test set is only used once at the end.

Single configuration from the command line:
    python run.py --layers 4 4 1 --acts tanh logistic --swarm 20 --iters 25 --runs 10
Learned hidden activations (extension):
    python run.py --layers 4 4 1 --learn_acts
For many configurations use sweep.py.
"""
import argparse
import csv
import numpy as np

from ann import ANN, ANNLearnedActs
from pso import PSO
from data import load_banknote, stratified_split, standardise

DEFAULTS = dict(layers=[4, 4, 1], acts=["tanh", "logistic"], learn_acts=False,
                swarm=20, iters=25, alpha=0.7, beta=1.5, gamma=1.5, delta=0.5,
                epsilon=1.0, informants=3, resample=False, range=2.0,
                boundary="clip", metric="mse")


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def make_fitness(ann, X, y, metric="mse"):
    """Build the fitness function PSO MINIMISES (the coupling point).
    Returns a training LOSS, so smaller = better."""
    def fitness(vec):
        ann.set_params(vec)
        if metric == "mse":                          # minimise squared error
            return float(np.mean((ann.forward(X)[:, 0] - y) ** 2))
        if metric == "bce":                          # minimise binary cross-entropy
            p = np.clip(ann.forward(X)[:, 0], 1e-7, 1 - 1e-7)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if metric == "error":                        # minimise 1 - accuracy
            return 1.0 - accuracy(y, ann.predict(X))
        raise ValueError(metric)
    return fitness


def prepare_split(path, split_seed=0):
    X, y = load_banknote(path)
    Xtr, ytr, Xte, yte = stratified_split(X, y, 0.3, split_seed)
    Xtr, Xte = standardise(Xtr, Xte)
    return Xtr, ytr, Xte, yte


def one_run(cfg, seed, split):
    """One PSO training run. cfg: dict like DEFAULTS. Returns a result dict."""
    cfg = {**DEFAULTS, **cfg}
    Xtr, ytr, Xte, yte = split
    if cfg["learn_acts"]:
        ann = ANNLearnedActs(cfg["layers"], cfg["acts"][-1], act_range=cfg["range"])
    else:
        ann = ANN(cfg["layers"], cfg["acts"])
    pso = PSO(dim=ann.n_params, fitness_fn=make_fitness(ann, Xtr, ytr, cfg["metric"]),
              swarm_size=cfg["swarm"], iterations=cfg["iters"],
              alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
              delta=cfg["delta"], epsilon=cfg["epsilon"],
              n_informants=cfg["informants"], pos_range=(-cfg["range"], cfg["range"]),
              boundary=cfg["boundary"], resample_informants=cfg["resample"], seed=seed)
    best_x, best_f, history, evals = pso.run()
    ann.set_params(best_x)
    return dict(train=accuracy(ytr, ann.predict(Xtr)), test=accuracy(yte, ann.predict(Xte)),
                evals=evals, history=[float(h) for h in history],
                acts=list(ann.activations))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_banknote_authentication.txt")
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULTS["layers"])
    p.add_argument("--acts", nargs="+", default=DEFAULTS["acts"])
    p.add_argument("--learn_acts", action="store_true",
                   help="encode hidden activations in the particle (extension)")
    for k in ["alpha", "beta", "gamma", "delta", "epsilon", "range"]:
        p.add_argument(f"--{k}", type=float, default=DEFAULTS[k])
    p.add_argument("--swarm", type=int, default=DEFAULTS["swarm"])
    p.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    p.add_argument("--informants", type=int, default=DEFAULTS["informants"])
    p.add_argument("--resample", action="store_true", help="re-draw informants each iteration")
    p.add_argument("--boundary", default="clip", choices=["none", "clip", "reflect"])
    p.add_argument("--metric", default="mse", choices=["mse", "bce", "error"])    
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV to append per-run results to")
    a = p.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}

    split = prepare_split(a.data, a.split_seed)
    rows = []
    for r in range(a.runs):
        res = one_run(cfg, r, split)
        rows.append((r, res))
        print(f"run {r}: train acc {res['train']:.4f}  test acc {res['test']:.4f}  "
              f"evals {res['evals']}" + (f"  acts {res['acts']}" if a.learn_acts else ""))
    tr = np.array([x[1]["train"] for x in rows]); te = np.array([x[1]["test"] for x in rows])
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"TRAIN mean {tr.mean():.4f} sd {sd(tr):.4f} | TEST mean {te.mean():.4f} sd {sd(te):.4f}")
    if a.out:
        with open(a.out, "a", newline="") as f:
            w = csv.writer(f)
            for r, res in rows:
                w.writerow([cfg, r, res["train"], res["test"], res["evals"]])


if __name__ == "__main__":
    main()
