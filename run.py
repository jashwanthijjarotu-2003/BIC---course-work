"""Task 3 + 4: couple PSO to the ANN (through the adapter) and train/evaluate.

Coupling: each PSO particle is a flat vector of length ann.n_params. The
MLPFitnessAdapter installs that vector into the ANN and returns the TRAINING LOSS,
which PSO minimises. The test set is only used once, at the end, to report metrics.

    python run.py --runs 10
    python run.py --layers 4 4 1 --learn_acts
    python run.py --loss bce --swarm 20 --iters 25
"""
import argparse
import numpy as np

from ann import ANN, ANNLearnedActs
from pso import PSO
from adapter import MLPFitnessAdapter
from metrics import binary_metrics
from data import load_banknote, stratified_split, standardise

DEFAULTS = dict(layers=[4, 4, 1], acts=["tanh", "logistic"], learn_acts=False,
                swarm=20, iters=25, alpha=0.7, beta=1.5, gamma=1.5, delta=0.5,
                epsilon=1.0, informants=3, resample=False, range=2.0,
                boundary="clip", metric="mse", informant_strategy="random")


def prepare_split(path, split_seed=0):
    X, y = load_banknote(path)
    Xtr, ytr, Xte, yte = stratified_split(X, y, 0.3, split_seed)
    Xtr, Xte = standardise(Xtr, Xte)
    return Xtr, ytr, Xte, yte


def one_run(cfg, seed, split):
    """One PSO training run. Returns a result dict (train loss + full metrics)."""
    cfg = {**DEFAULTS, **cfg}
    Xtr, ytr, Xte, yte = split

    # build the ANN
    if cfg["learn_acts"]:
        ann = ANNLearnedActs(cfg["layers"], cfg["acts"][-1], act_range=cfg["range"])
    else:
        ann = ANN(cfg["layers"], cfg["acts"])

    # the adapter turns the ANN into a FitnessFunction PSO can minimise (training loss)
    fitness = MLPFitnessAdapter(ann, Xtr, ytr, cfg["metric"])
    pso = PSO(dim=ann.n_params, fitness_fn=fitness,
              swarm_size=cfg["swarm"], iterations=cfg["iters"],
              alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
              delta=cfg["delta"], epsilon=cfg["epsilon"],
              n_informants=cfg["informants"], pos_range=(-cfg["range"], cfg["range"]),
              boundary=cfg["boundary"], resample_informants=cfg["resample"],
              informant_strategy=cfg["informant_strategy"], seed=seed)
    best_x, best_loss, history, evals = pso.run()

    # install the best weights, then measure full metrics on train and unseen test
    ann.set_params(best_x)
    tr = binary_metrics(ytr, ann.predict(Xtr))
    te = binary_metrics(yte, ann.predict(Xte))
    return dict(train=tr["accuracy"], test=te["accuracy"], train_loss=float(best_loss),
                precision=te["precision"], recall=te["recall"], f1=te["f1"],
                evals=evals, history=[float(h) for h in history],
                acts=[repr(a) for a in ann.activations])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_banknote_authentication.txt")
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULTS["layers"])
    p.add_argument("--acts", nargs="+", default=DEFAULTS["acts"])
    p.add_argument("--learn_acts", action="store_true")
    p.add_argument("--loss", dest="metric", default="mse", choices=["mse", "bce", "error"])
    for k in ["alpha", "beta", "gamma", "delta", "epsilon", "range"]:
        p.add_argument(f"--{k}", type=float, default=DEFAULTS[k])
    p.add_argument("--swarm", type=int, default=DEFAULTS["swarm"])
    p.add_argument("--iters", type=int, default=DEFAULTS["iters"])
    p.add_argument("--informants", type=int, default=DEFAULTS["informants"])
    p.add_argument("--resample", action="store_true")
    p.add_argument("--boundary", default="clip", choices=["none", "clip", "reflect"])
    p.add_argument("--informant_strategy", default="random", choices=["random", "nearest"])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--split_seed", type=int, default=0)
    a = p.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}

    split = prepare_split(a.data, a.split_seed)
    rows = [one_run(cfg, r, split) for r in range(a.runs)]
    for r, res in enumerate(rows):
        print(f"run {r}: train_loss {res['train_loss']:.4f}  test_acc {res['test']:.4f}  "
              f"P {res['precision']:.3f}  R {res['recall']:.3f}  F1 {res['f1']:.3f}")

    def ms(key):
        v = np.array([res[key] for res in rows])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    print("-" * 62)
    for key, lab in [("train_loss", "best train loss"), ("test", "test accuracy"),
                     ("precision", "precision"), ("recall", "recall"), ("f1", "F1")]:
        m, s = ms(key)
        print(f"{lab:16s}: {m:.4f} +/- {s:.4f}")


if __name__ == "__main__":
    main()
