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
        expected = getattr(self, "n_weight_params", self.n_params)  # subclass adds extra genes
        if vec.size != expected:
            raise ValueError(f"Expected {expected} params, got {vec.size}")
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

    def predict(self, X, threshold=None):
        """Binary class labels from a single output node.
        threshold=None picks 0.0 for a tanh output and 0.5 otherwise."""
        if threshold is None:
            threshold = 0.0 if self.activations[-1] == "tanh" else 0.5
        return (self.forward(X)[:, 0] >= threshold).astype(int)


class ANNLearnedActs(ANN):
    """EXTENSION: the hidden-layer activation functions are part of the PSO vector.

    Particle layout: [all weights and biases ... | one gene per hidden layer].
    Each extra gene g in [-r, r] is decoded to logistic / relu / tanh by splitting
    [-r, r] into three equal bins. The output layer's activation is fixed.
    """
    CHOICES = ["logistic", "relu", "tanh"]

    def __init__(self, layer_sizes, output_activation="logistic", act_range=2.0):
        self._ready = False                     # lets the base __init__ call set_params
        self.act_range = act_range
        self.n_hidden = len(layer_sizes) - 2
        super().__init__(layer_sizes, ["tanh"] * self.n_hidden + [output_activation])
        self._ready = True
        self.n_weight_params = self.n_params
        self.n_params += self.n_hidden
        self.set_params(np.zeros(self.n_params))

    def decode_act(self, gene):
        r = self.act_range
        frac = (np.clip(gene, -r, r) + r) / (2 * r)
        return self.CHOICES[min(int(frac * len(self.CHOICES)), len(self.CHOICES) - 1)]

    def set_params(self, vec):
        vec = np.asarray(vec, dtype=float)
        if not self._ready:
            return super().set_params(vec)
        if vec.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {vec.size}")
        super().set_params(vec[:self.n_weight_params])
        for layer, gene in enumerate(vec[self.n_weight_params:]):
            self.activations[layer] = self.decode_act(gene)
