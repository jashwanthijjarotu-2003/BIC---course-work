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

Fitness is MINIMISED (lower = better): the fitness function returns a training
loss, so smaller is better. (Algorithm 39 is written for maximisation; we keep
the same structure but compare with '<'.)

Informants are chosen through an interchangeable STRATEGY (random or nearest-neighbour).
"""
import numpy as np


class Particle:
    """One particle: its own position, velocity, personal best, and informants."""
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()    # personal best location
        self.best_fitness = np.inf              # personal best score (minimise -> start high)
        self.fitness = np.inf                   # current score
        self.informants = []                    # list of other Particle objects (+ self)

    def evaluate(self, fitness_fn):
        """Score this particle's current position."""
        self.fitness = fitness_fn(self.position)
        return self.fitness

    def update_personal_best(self):
        """Keep this position if it is the best (lowest) this particle has seen."""
        if self.fitness < self.best_fitness:    # '<' because we MINIMISE
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

    def get_informants_best(self):              # Alg39 L18: x+ (informants' best, incl. self)
        """The best personal-best position among this particle's informants."""
        best = min(self.informants, key=lambda p: p.best_fitness)
        return best.best_position

    def update_velocity(self, informants_best, global_best,
                        alpha, beta, gamma, delta, rng):   # Alg39 L16-24
        D = len(self.position)
        b = rng.uniform(0, beta, D)             # Alg39 L21
        c = rng.uniform(0, gamma, D)            # Alg39 L22
        d = rng.uniform(0, delta, D)            # Alg39 L23
        self.velocity = (alpha * self.velocity                          # Alg39 L24
                         + b * (self.best_position - self.position)     #   x* own best   (L17)
                         + c * (informants_best    - self.position)     #   x+ informants (L18)
                         + d * (global_best        - self.position))    #   x! global     (L19)

    def update_position(self, epsilon):         # Alg39 L25-26
        self.position = self.position + epsilon * self.velocity


# ---- informant selection (Strategy pattern) ---------------------------------
class InformantSelection:
    """Abstract strategy: choose a particle's informants (always includes self)."""
    def select(self, particle, particles, k, rng):
        raise NotImplementedError


class RandomInformants(InformantSelection):
    """k random OTHER particles, plus the particle itself."""
    def select(self, particle, particles, k, rng):
        others = [p for p in particles if p is not particle]
        idx = rng.choice(len(others), size=k, replace=False)
        return [others[i] for i in idx] + [particle]


class NearestInformants(InformantSelection):
    """the k CLOSEST other particles by Euclidean distance, plus the particle itself."""
    def select(self, particle, particles, k, rng):
        others = [p for p in particles if p is not particle]
        dist = [np.sum((p.position - particle.position) ** 2) for p in others]  # squared Euclidean
        order = np.argsort(dist)[:k]
        return [others[i] for i in order] + [particle]


def make_informant_strategy(name):
    if name == "random":
        return RandomInformants()
    if name == "nearest":
        return NearestInformants()
    raise ValueError(f"Unknown informant strategy '{name}' (use 'random' or 'nearest')")


class PSO:
    def __init__(self, dim, fitness_fn, swarm_size=30, iterations=100,
                 alpha=0.7, beta=1.5, gamma=1.5, delta=0.5, epsilon=1.0,
                 n_informants=3, pos_range=(-1.0, 1.0), boundary="clip",
                 resample_informants=False, informant_strategy="random", seed=None):
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
        self.informant_strategy = informant_strategy          # "random" | "nearest"
        self.strategy = make_informant_strategy(informant_strategy)
        self.rng = np.random.default_rng(seed)

    # ---- setup --------------------------------------------------------------
    def _make_particles(self):
        """Alg39 L7-9: create random particles with random initial velocities."""
        span = self.hi - self.lo
        particles = []
        for _ in range(self.swarm_size):                        # Alg39 L8
            pos = self.rng.uniform(self.lo, self.hi, self.dim)  # Alg39 L9 (random location)
            vel = self.rng.uniform(-0.1 * span, 0.1 * span, self.dim)
            particles.append(Particle(pos, vel))
        return particles

    def _assign_informants(self, particles):
        """Use the chosen strategy to give each particle its informants (+ itself)."""
        k = min(self.n_informants, self.swarm_size - 1)
        for p in particles:
            p.informants = self.strategy.select(p, particles, k, self.rng)

    def _handle_boundary(self, p):              # extension (not in Alg 39)
        if self.boundary == "none":
            return
        if self.boundary == "clip":             # stick to wall, kill velocity there
            out = (p.position < self.lo) | (p.position > self.hi)
            p.position = np.clip(p.position, self.lo, self.hi)
            p.velocity = np.where(out, 0.0, p.velocity)
        elif self.boundary == "reflect":        # bounce back, reverse velocity
            for _ in range(2):  # handles overshoot up to ~1 span
                over, under = p.position > self.hi, p.position < self.lo
                p.position = np.where(over, 2 * self.hi - p.position, p.position)
                p.position = np.where(under, 2 * self.lo - p.position, p.position)
                p.velocity = np.where(over | under, -p.velocity, p.velocity)
            p.position = np.clip(p.position, self.lo, self.hi)
        else:
            raise ValueError(f"Unknown boundary handling: {self.boundary}")

    # ---- main loop ----------------------------------------------------------
    def run(self):
        particles = self._make_particles()      # Alg39 L7-9
        self._assign_informants(particles)
        best_x, best_f = None, np.inf           # Alg39 L10: Best <- none (minimise -> +inf)
        history, evals = [], 0

        for _ in range(self.iterations):        # Alg39 L11 (repeat ... until time out L27)
            # Alg39 L12-15: assess each particle; update personal best and global Best
            for p in particles:
                p.evaluate(self.fitness_fn)
                evals += 1
                p.update_personal_best()
                if p.best_fitness < best_f:     # minimise -> keep the smaller loss
                    best_f, best_x = p.best_fitness, p.best_position.copy()
            history.append(best_f)

            # nearest-neighbour informants depend on current positions, so reassign each
            # iteration; random informants only reassign if resample is on.
            if self.resample_informants or self.informant_strategy == "nearest":
                self._assign_informants(particles)

            # Alg39 L16-24: velocity update
            for p in particles:
                informants_best = p.get_informants_best()               # Alg39 L18: x+
                p.update_velocity(informants_best, best_x,              # L17 x*, L19 x!
                                  self.alpha, self.beta, self.gamma, self.delta, self.rng)

            # Alg39 L25-26: move particles
            for p in particles:
                p.update_position(self.epsilon)
                self._handle_boundary(p)        # extension (not in Alg 39)

        return best_x, best_f, history, evals   # Alg39 L28


# ---------------------------------------------------------------------------
# INDEPENDENCE DEMO: PSO knows nothing about ANNs. Run `python pso.py` to see it
# minimise the sphere function f(x)=sum(x^2) (minimum 0 at the origin).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def sphere(v):
        return float(np.sum(v ** 2))            # a loss: minimise -> go to 0

    pso = PSO(dim=2, fitness_fn=sphere, swarm_size=30, iterations=100,
              pos_range=(-5.0, 5.0), seed=0)
    best_x, best_f, history, evals = pso.run()
    print("Sphere benchmark (no ANN involved):")
    print("  best position :", best_x)          # expect ~ [0, 0]
    print("  best f value  :", best_f)           # expect ~ 0
    print("  evaluations   :", evals)
