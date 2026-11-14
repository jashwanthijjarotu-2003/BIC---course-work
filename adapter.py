"""Adapter pattern: connect the MLP to the PSO through an abstract FitnessFunction.

PSO only needs something it can call with a position vector to get back a scalar to
MINIMISE. It knows nothing about MLPs. The adapter is the ONLY place that knows how to
turn a parameter vector into an MLP training loss.
"""
import numpy as np


class FitnessFunction:
    """Abstract interface that PSO optimises. Subclasses implement evaluate()."""
    def evaluate(self, position):
        """Take a candidate solution vector, return a scalar fitness (to minimise)."""
        raise NotImplementedError

    def __call__(self, position):
        # lets PSO call the object like a plain function: fitness(position)
        return self.evaluate(position)


class MLPFitnessAdapter(FitnessFunction):
    """Makes an MLP + a loss look like a FitnessFunction to PSO.

    evaluate(position):
        1. install the parameter vector in the network,
        2. forward pass on the TRAINING data,
        3. return the training LOSS (a scalar PSO minimises).
    Only training data is used here (never the test set).
    """
    def __init__(self, ann, X_train, y_train, metric="mse"):
        self.ann = ann
        self.X = X_train
        self.y = y_train
        self.metric = metric

    def evaluate(self, position):
        self.ann.set_params(position)                 # install candidate weights
        if self.metric == "mse":                      # mean squared error
            out = self.ann.forward(self.X)[:, 0]
            return float(np.mean((out - self.y) ** 2))
        if self.metric == "bce":                      # binary cross-entropy
            out = self.ann.forward(self.X)[:, 0]
            p = np.clip(out, 1e-7, 1 - 1e-7)
            return float(-np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))
        if self.metric == "error":                    # misclassification rate (1 - accuracy)
            return float(np.mean(self.ann.predict(self.X) != self.y))
        raise ValueError(f"Unknown metric '{self.metric}'")
