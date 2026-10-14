"""Task 1: Feedforward multi-layer ANN, implemented from scratch (NumPy only for maths).

No training algorithm lives here. The ANN exposes its weights and biases as ONE flat
vector (get/set_params) so that PSO can optimise them (see run.py for the coupling).
"""
import numpy as np


# ---- Activation functions (required: logistic, ReLU, tanh) -------------------
def logistic(x):
    # clip to avoid overflow in exp for large |x|
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0.0, x)


def tanh(x):
    return np.tanh(x)


ACTIVATIONS = {"logistic": logistic, "relu": relu, "tanh": tanh}


class ANN:
    """Fully connected feedforward network.

    layer_sizes: e.g. [4, 8, 1]  -> 4 inputs, one hidden layer of 8, 1 output
    activations: one name per NON-input layer, e.g. ["tanh", "logistic"]
    """

    def __init__(self, layer_sizes, activations):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Need one activation per non-input layer")
        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        # Parameter count: for each layer, (n_in * n_out) weights + n_out biases
        self.n_params = sum(
            n_in * n_out + n_out
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        self.weights = []
        self.biases = []
        self.set_params(np.zeros(self.n_params))

    def set_params(self, vec):
        """Unpack a flat vector into per-layer weight matrices and bias vectors."""
        vec = np.asarray(vec, dtype=float)
        if vec.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {vec.size}")
        self.weights, self.biases = [], []
        pos = 0
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = vec[pos:pos + n_in * n_out].reshape(n_in, n_out)
            pos += n_in * n_out
            b = vec[pos:pos + n_out]
            pos += n_out
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """X: (n_samples, n_features) -> (n_samples, n_outputs)."""
        a = np.asarray(X, dtype=float)
        for w, b, act in zip(self.weights, self.biases, self.activations):
            a = ACTIVATIONS[act](a @ w + b)
        return a

    def predict(self, X, threshold=0.5):
        """Binary class labels. Assumes a single output node whose range suits a
        0.5 threshold (logistic). For tanh outputs use threshold=0.0."""
        return (self.forward(X)[:, 0] >= threshold).astype(int)
