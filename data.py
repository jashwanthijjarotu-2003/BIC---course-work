"""Task 4 data handling: load banknote CSV, stratified 70/30 split, standardise."""
import numpy as np


def load_banknote(path="data_banknote_authentication.txt"):
    """Rows: 4 features, then label (0/1). Comma separated, no header."""
    data = np.loadtxt(path, delimiter=",")
    return data[:, :4], data[:, 4].astype(int)


def stratified_split(X, y, test_frac=0.3, seed=0):
    """Keeps the class ratio the same in train and test."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_test = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx, test_idx = rng.permutation(train_idx), rng.permutation(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardise(X_train, X_test):
    """z-score using TRAIN statistics only (avoids test-set leakage)."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd
