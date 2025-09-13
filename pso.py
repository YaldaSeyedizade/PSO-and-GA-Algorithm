import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, func, dim, bounds, pop_size=50, w=0.74, c1=1.42, c2=1.42, budget=40000, seed=None):
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.budget = budget
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.iters = budget // pop_size

    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def run(self):
        X = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        V = self.rng.uniform(-abs(self.ub - self.lb), abs(self.ub - self.lb), size=(self.pop_size, self.dim)) * 0.1
        P = X.copy()
        P_val = np.array([self.func(x) for x in X])
        G = P[np.argmin(P_val)].copy()
        G_val = np.min(P_val)

        for _ in range(self.iters):
            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random((self.pop_size, self.dim))
            V = self.w*V + self.c1*r1*(P - X) + self.c2*r2*(G - X)
            X = self.clip(X + V)
            vals = np.array([self.func(x) for x in X])
            improved = vals < P_val
            P[improved] = X[improved]
            P_val[improved] = vals[improved]
            if np.min(P_val) < G_val:
                G = P[np.argmin(P_val)].copy()
                G_val = np.min(P_val)

        return G_val
