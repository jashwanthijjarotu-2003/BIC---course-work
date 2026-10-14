"""Task 3 + 4: couple PSO to the ANN and train/evaluate on the banknote data.

Coupling: each PSO particle is a flat vector of length ann.n_params. To evaluate a
particle we ann.set_params(particle) and measure the ANN's accuracy on the TRAINING set;
that accuracy is the particle's fitness. The test set is only used once at the end.

Usage example (one configuration, 10 repeated runs):
    python run.py --layers 4 8 1 --acts tanh logistic --swarm 30 --iters 100 \
                  --beta 1.5 --gamma 1.5 --delta 0.5 --runs 10 --out results.csv
"""
import argparse
import csv
import numpy as np

from ann import ANN
from pso import PSO
from data import load_banknote, stratified_split, standardise


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def make_fitness(ann, X, y, metric="accuracy"):
    """Build the fitness function PSO maximises (the coupling point)."""
    def fitness(vec):
        ann.set_params(vec)
        if metric == "accuracy":
            return accuracy(y, ann.predict(X))
        if metric == "neg_mse":                 # smoother landscape than accuracy
            return -float(np.mean((ann.forward(X)[:, 0] - y) ** 2))
        raise ValueError(metric)
    return fitness


def one_run(args, seed, split):
    Xtr, ytr, Xte, yte = split
    ann = ANN(args.layers, args.acts)
    pso = PSO(dim=ann.n_params, fitness_fn=make_fitness(ann, Xtr, ytr, args.metric),
              swarm_size=args.swarm, iterations=args.iters,
              alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
              epsilon=args.epsilon, n_informants=args.informants,
              pos_range=(-args.range, args.range), boundary=args.boundary, seed=seed)
    best_x, best_f, history, evals = pso.run()
    ann.set_params(best_x)
    return accuracy(ytr, ann.predict(Xtr)), accuracy(yte, ann.predict(Xte)), evals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_banknote_authentication.txt")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 1])
    p.add_argument("--acts", nargs="+", default=["tanh", "logistic"])
    p.add_argument("--swarm", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--informants", type=int, default=3)
    p.add_argument("--range", type=float, default=2.0, help="weights init/bounds in [-r, r]")
    p.add_argument("--boundary", default="clip", choices=["none", "clip", "reflect"])
    p.add_argument("--metric", default="accuracy", choices=["accuracy", "neg_mse"])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional CSV to append per-run results to")
    args = p.parse_args()

    X, y = load_banknote(args.data)
    Xtr, ytr, Xte, yte = stratified_split(X, y, 0.3, args.split_seed)
    Xtr, Xte = standardise(Xtr, Xte)
    split = (Xtr, ytr, Xte, yte)

    rows = []
    for r in range(args.runs):
        tr, te, evals = one_run(args, seed=r, split=split)
        rows.append((r, tr, te, evals))
        print(f"run {r}: train acc {tr:.4f}  test acc {te:.4f}  evals {evals}")
    tr_a, te_a = np.array([x[1] for x in rows]), np.array([x[2] for x in rows])
    print(f"TRAIN mean {tr_a.mean():.4f} sd {tr_a.std(ddof=1) if len(tr_a)>1 else 0:.4f} | "
          f"TEST mean {te_a.mean():.4f} sd {te_a.std(ddof=1) if len(te_a)>1 else 0:.4f}")

    if args.out:
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            for r, tr, te, evals in rows:
                w.writerow([vars(args)["layers"], args.acts, args.swarm, args.iters,
                            args.alpha, args.beta, args.gamma, args.delta, args.epsilon,
                            args.informants, args.boundary, args.metric, r, tr, te, evals])


if __name__ == "__main__":
    main()
