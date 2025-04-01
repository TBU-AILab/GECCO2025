import random

class DifferentialEvolution:
    def __init__(self, func, dim, bounds, max_evals, population_factor=20, F=0.8, CR=0.9):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.population_factor = population_factor
        self.F = F
        self.CR = CR
        self.population_size = population_factor * dim
        self.evals = 0

        # Initialize population
        self.population = [
            [random.uniform(low, high) for low, high in self.bounds]
            for _ in range(self.population_size)
        ]
        self.fitnesses = [self.evaluate(ind) for ind in self.population]

        # Track the best solution
        self.best_solution = None
        self.best_fitness = float('inf')
        self.update_best()

    def evaluate(self, individual):
        fitness = self.func(individual)
        self.evals += 1
        return fitness

    def update_best(self):
        for ind, fit in zip(self.population, self.fitnesses):
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_solution = ind[:]

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = random.sample(indices, 3)
        y = [
            self.population[a][j] + self.F * (self.population[b][j] - self.population[c][j])
            for j in range(self.dim)
        ]
        return [max(min(y[j], self.bounds[j][1]), self.bounds[j][0]) for j in range(self.dim)]

    def crossover(self, target, mutant):
        return [
            mutant[j] if random.random() < self.CR or j == random.randint(0, self.dim - 1) else target[j]
            for j in range(self.dim)
        ]

    def optimize(self):
        while self.evals < self.max_evals:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)

                trial_fitness = self.evaluate(trial)

                # Replace and update best solution
                if trial_fitness < self.fitnesses[i]:
                    self.population[i] = trial
                    self.fitnesses[i] = trial_fitness

                self.update_best()

                if self.evals >= self.max_evals:
                    break

        return self.best_fitness, self.best_solution

# Example usage
# def objective_function(x):
#     return sum([xi ** 2 for xi in x])
# dim = 10
# bounds = [(-5, 5) for _ in range(dim)]
# max_evals = 1000
# de = DifferentialEvolution(objective_function, dim, bounds, max_evals)
# best_cost, best_solution = de.optimize()
# print(f"Best cost: {best_cost}, Best solution: {best_solution}")
