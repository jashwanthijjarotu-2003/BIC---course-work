"""Task 2: Particle Swarm Optimisation with informants, from scratch.

Follows Algorithm 39 (Section 3.5, "Essentials of Metaheuristics", Luke).
Comments tag the corresponding algorithm line numbers as "Alg39 L<n>".

Algorithm 39 (as used here):
 1 swarmsize   2 alpha (velocity retained)   3 beta (personal best)
 4 gamma (informants' best)   5 delta (global best)   6 epsilon (jump size)
 7 P <- {}   8-9 create random particles with random velocity   10 Best <- none
 11 repeat
 12-15   assess each particle; update Best
 16-24   for each particle: find x* (own best), x+ (informants' best, incl. self),
         x! (global best); for each dimension: draw b,c,d and update velocity
 25-26   move each particle: x <- x + epsilon * v
 27 until Best ideal or out of time        28 return Best

Fitness is MAXIMISED (higher = better), matching Algorithm 39.
"""
import numpy as np


class PSO:
    def __init__(self, dim, fitness_fn, swarm_size=30, iterations=100,
                 alpha=0.7, beta=1.5, gamma=1.5, delta=0.5, epsilon=1.0,
                 n_informants=3, pos_range=(-1.0, 1.0), boundary="clip",
                 resample_informants=False, seed=None):
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.swarm_size = swarm_size            # Alg39 L1
        self.iterations = iterations
        self.alpha, self.beta = alpha, beta     # Alg39 L2, L3
        self.gamma, self.delta = gamma, delta   # Alg39 L4, L5
        self.epsilon = epsilon                  # Alg39 L6
        self.n_informants = n_informants        # informants per particle (excl. self)
        self.lo, self.hi = pos_range
        self.boundary = boundary                # "none" | "clip" | "reflect"
        self.resample_informants = resample_informants
        self.rng = np.random.default_rng(seed)

    # ---- informant topology -------------------------------------------------
    def _make_informants(self):
        """Each particle is informed by itself + n_informants random others."""
        S = self.swarm_size
        k = min(self.n_informants, S - 1)
        inf = []
        for i in range(S):
            others = [j for j in range(S) if j != i]
            chosen = self.rng.choice(others, size=k, replace=False)
            inf.append(np.append(chosen, i))
        return inf

    def _handle_boundary(self, X, V):
        if self.boundary == "none":
            return X, V
        if self.boundary == "clip":             # stick to wall, kill velocity there
            out = (X < self.lo) | (X > self.hi)
            X = np.clip(X, self.lo, self.hi)
            V = np.where(out, 0.0, V)
        elif self.boundary == "reflect":        # bounce back, reverse velocity
            span = self.hi - self.lo
            for _ in range(2):  # handles overshoot up to ~1 span
                over, under = X > self.hi, X < self.lo
                X = np.where(over, 2 * self.hi - X, X)
                X = np.where(under, 2 * self.lo - X, X)
                V = np.where(over | under, -V, V)
            X = np.clip(X, self.lo, self.hi)
        else:
            raise ValueError(f"Unknown boundary handling: {self.boundary}")
        return X, V

    def run(self):
        S, D = self.swarm_size, self.dim
        span = self.hi - self.lo

        # Alg39 L7-9: random particles with random initial velocities
        X = self.rng.uniform(self.lo, self.hi, (S, D))
        V = self.rng.uniform(-0.1 * span, 0.1 * span, (S, D))

        pbest_X = X.copy()                      # "previous fittest location" of each particle
        pbest_f = np.full(S, -np.inf)
        best_x, best_f = None, -np.inf          # Alg39 L10: Best <- none
        informants = self._make_informants()
        history = []                            # best fitness after each iteration
        evals = 0

        for _ in range(self.iterations):        # Alg39 L11 (repeat ... until time out L27)
            # Alg39 L12-15: assess fitness, track Best (and each particle's own best)
            for i in range(S):
                f = self.fitness_fn(X[i])
                evals += 1
                if f > pbest_f[i]:
                    pbest_f[i], pbest_X[i] = f, X[i].copy()
                if f > best_f:
                    best_f, best_x = f, X[i].copy()
            history.append(best_f)

            if self.resample_informants:
                informants = self._make_informants()

            # Alg39 L16-24: velocity update
            g_idx = int(np.argmax(pbest_f))
            x_global = pbest_X[g_idx]           # Alg39 L19: x! (best of any particle)
            newV = np.empty_like(V)
            for i in range(S):
                x_star = pbest_X[i]             # Alg39 L17: x* own best
                inf = informants[i]
                x_plus = pbest_X[inf[np.argmax(pbest_f[inf])]]  # Alg39 L18: x+
                b = self.rng.uniform(0, self.beta, D)    # Alg39 L21
                c = self.rng.uniform(0, self.gamma, D)   # Alg39 L22
                d = self.rng.uniform(0, self.delta, D)   # Alg39 L23
                newV[i] = (self.alpha * V[i]             # Alg39 L24
                           + b * (x_star - X[i])
                           + c * (x_plus - X[i])
                           + d * (x_global - X[i]))
            V = newV

            # Alg39 L25-26: move particles
            X = X + self.epsilon * V
            X, V = self._handle_boundary(X, V)  # extension (not in Alg 39)

        return best_x, best_f, history, evals   # Alg39 L28
