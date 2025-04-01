import random

class DifferentialEvolution:
    def __init__(self, dim, bounds, func, max_evals):
        self.dim = dim
        self.bounds = bounds
        self.func = func
        self.max_evals = max_evals
        self.evals = 0
        self.population_size = 100
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = []
        self.fitness_population = []

    def create_individual(self):
        return [random.uniform(low, high) for low, high in self.bounds]

    def mutate(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = random.sample(candidates, 3)
        mutant = [
            max(min(self.bounds[i][1], self.bounds[i][0] + self.mutation_factor * (self.population[a][i] - self.population[b][i] + self.population[c][i])), self.bounds[i][0])
            for i in range(self.dim)
        ]
        return mutant

    def crossover(self, target, mutant):
        return [
            mutant[i] if random.random() < self.crossover_rate else target[i]
            for i in range(self.dim)
        ]

    def optimize(self):
        self.population = [self.create_individual() for _ in range(self.population_size)]
        self.fitness_population = [self.evaluate(ind) for ind in self.population]

        while self.evals < self.max_evals:
            for idx in range(self.population_size):
                mutate_vector = self.mutate(idx)
                trial_vector = self.crossover(self.population[idx], mutate_vector)
                trial_fitness = self.evaluate(trial_vector)

                if trial_fitness < self.fitness_population[idx]:
                    self.population[idx] = trial_vector
                    self.fitness_population[idx] = trial_fitness

        return min(self.fitness_population)

    def evaluate(self, individual):
        self.evals += 1
        return self.func(individual)

def run_algorithm(func, dim, bounds, max_evals):
    optimizer = DifferentialEvolution(dim, bounds, func, max_evals)
    best_result = optimizer.optimize()
    return best_result

# Example usage with a test function
def test_function(x):
    return sum(xi ** 2 for xi in x)

if __name__ == "__main__":
    dim = 10
    bounds = [(-100, 100)] * dim
    max_evals = 10000
    best_result = run_algorithm(test_function, dim, bounds, max_evals)
    print("Best result found:", best_result)
#
#I've removed the `run` function and directly called the `run_algorithm` function within the `__main__` block. This ensures that there is no `KeyError('run')` because we are using function names directly consistent with their use in Python scripts.
