import numpy as np

class GeneticAlgorithm:
    def __init__(self, func, dim, bounds, pop_size=50, pc=0.75, pm=0.01, budget=40000, seed=None):
        self.func = func
        self.dim = dim
        self.lb, self.ub = bounds
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.budget = budget
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.iters = budget // pop_size

    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def fitness(self, pop):
        return np.array([self.func(ind) for ind in pop])

    def tournament(self, pop, fit):
        idx = self.rng.integers(0, self.pop_size, size=2)
        return pop[idx[np.argmin(fit[idx])]]

    def crossover(self, p1, p2):
        u = self.rng.random(self.dim)
        beta = np.where(u <= 0.5, (2*u)**(1/16), (1/(2*(1-u)))**(1/16))
        c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
        c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
        return self.clip(c1), self.clip(c2)

    def mutate(self, x):
        return self.clip(x + self.rng.normal(0, 0.1*(self.ub - self.lb), size=self.dim))

    def run(self):
        pop = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fit = self.fitness(pop)
        best = np.min(fit)

        for _ in range(self.iters):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1, p2 = self.tournament(pop, fit), self.tournament(pop, fit)
                c1, c2 = p1.copy(), p2.copy()
                if self.rng.random() < self.pc:
                    c1, c2 = self.crossover(p1, p2)
                if self.rng.random() < self.pm:
                    c1 = self.mutate(c1)
                if self.rng.random() < self.pm:
                    c2 = self.mutate(c2)
                new_pop.extend([c1, c2])
            pop = np.array(new_pop[:self.pop_size])
            fit = self.fitness(pop)
            best = min(best, np.min(fit))

        return best
