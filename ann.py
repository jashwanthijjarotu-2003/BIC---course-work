"""Task 1: Feedforward multi-layer ANN, implemented from scratch (NumPy only for maths).

No training algorithm lives here. The ANN exposes its weights and biases as ONE flat
vector (get_params / set_params) so that PSO can optimise them (see run.py for coupling).

Each layer uses an ACTIVATION-FUNCTION OBJECT (an ActivationFunction subclass with a
forward() method)..
"""
import numpy as np


# ---- Activation functions as OBJECTS -----------------------------------------
class ActivationFunction:
    """Abstract base: every activation implements forward(x)."""
    def forward(self, x):
        raise NotImplementedError

    def __repr__(self):                      # so printing shows e.g. "Tanh"
        return self.__class__.__name__


class Logistic(ActivationFunction):
    """Logistic / sigmoid -> output in (0, 1)."""
    def forward(self, x):
        # clip to avoid overflow in exp for large |x|
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class ReLU(ActivationFunction):
    """Rectified linear -> max(0, x)."""
    def forward(self, x):
        return np.maximum(0.0, x)


class Tanh(ActivationFunction):
    """Hyperbolic tangent -> output in (-1, 1)."""
    def forward(self, x):
        return np.tanh(x)


# name -> class, so the network can be configured with simple strings
ACTIVATIONS = {"logistic": Logistic, "relu": ReLU, "tanh": Tanh}


def make_activation(a):
    """Turn a name ('tanh') OR an existing object into an activation object."""
    if isinstance(a, ActivationFunction):    # already an object -> use as is
        return a
    key = str(a).lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{a}'. Options: {list(ACTIVATIONS)}")
    return ACTIVATIONS[key]()                # build a new object of that class


class ANN:
    """Fully connected feedforward network.

    layer_sizes: e.g. [4, 8, 1]  -> 4 inputs, one hidden layer of 8, 1 output
    activations: one per NON-input layer, given as names or objects, e.g. ["tanh", "logistic"]
    """

    def __init__(self, layer_sizes, activations):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Need one activation per non-input layer")
        self.layer_sizes = list(layer_sizes)
        # store activation OBJECTS (convert any names into objects)
        self.activations = [make_activation(a) for a in activations]
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

    def get_params(self):
        """Pack all weight matrices and bias vectors back into ONE flat vector.
        The inverse of set_params; same ordering (weights then biases, layer by layer)."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b)
        return np.concatenate(parts)

    def forward(self, X):
        """X: (n_samples, n_features) -> (n_samples, n_outputs)."""
        a = np.asarray(X, dtype=float)
        for w, b, act in zip(self.weights, self.biases, self.activations):
            a = act.forward(a @ w + b)           # call the activation OBJECT's forward()
        return a

    def predict(self, X, threshold=None):
        """Binary class labels from a single output node.
        threshold=None picks 0.0 for a tanh output and 0.5 otherwise."""
        if threshold is None:
            threshold = 0.0 if isinstance(self.activations[-1], Tanh) else 0.5
        return (self.forward(X)[:, 0] >= threshold).astype(int)


class ANNLearnedActs(ANN):
    """EXTENSION: the hidden-layer activation functions are part of the PSO vector.

    Particle layout: [all weights and biases ... | one gene per hidden layer].
    Each extra gene g in [-r, r] is decoded to a logistic / relu / tanh OBJECT by
    splitting [-r, r] into three equal bins. The output layer's activation is fixed.
    """
    CHOICES = ["logistic", "relu", "tanh"]

    def __init__(self, layer_sizes, output_activation="logistic", act_range=2.0):
        self._ready = False
        self.act_range = act_range
        self.n_hidden = len(layer_sizes) - 2
        super().__init__(layer_sizes, ["tanh"] * self.n_hidden + [output_activation])
        self._ready = True
        self.n_weight_params = self.n_params
        self.n_params += self.n_hidden
        self.set_params(np.zeros(self.n_params))

    def decode_act(self, gene):
        """One gene -> one activation OBJECT."""
        r = self.act_range
        frac = (np.clip(gene, -r, r) + r) / (2 * r)
        name = self.CHOICES[min(int(frac * len(self.CHOICES)), len(self.CHOICES) - 1)]
        return make_activation(name)

    def set_params(self, vec):
        vec = np.asarray(vec, dtype=float)
        if not self._ready:
            return super().set_params(vec)
        if vec.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {vec.size}")
        super().set_params(vec[:self.n_weight_params])
        for layer, gene in enumerate(vec[self.n_weight_params:]):
            self.activations[layer] = self.decode_act(gene)   # store the object
