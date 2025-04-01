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
        
def run_optimizer(func, dim, bounds, max_evals):
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
    best_result = run_optimizer(test_function, dim, bounds, max_evals)
    print("Best result found:", best_result)
#
#This version includes a complete run example with a sample test function (`test_function`), which is a simple sphere function. The error related to `KeyError('run')` was addressed by ensuring that the `run_optimizer` function is correctly defined and invoked within the `__main__` section. This callable structure sets up the algorithm with clearly defined arguments, such as function, bounds, and the number of evaluations, allowing easy integration into testing frameworks or execution environments.
