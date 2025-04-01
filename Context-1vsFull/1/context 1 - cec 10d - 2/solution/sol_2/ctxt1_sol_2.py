import random

def run(func, dim, bounds, max_evals):
    # Optimizer configuration
    population_size = 40
    mutation_factor = 0.8
    crossover_rate = 0.9
    evals = 0

    def create_individual():
        return [random.uniform(low, high) for low, high in bounds]

    def mutate(candidate, population):
        candidates = list(range(population_size))
        candidates.remove(candidate)
        a, b, c = random.sample(candidates, 3)
        return [
            max(min(bounds[i][1], bounds[i][0] + mutation_factor * (population[a][i] - population[b][i] + population[c][i])), bounds[i][0])
            for i in range(dim)
        ]

    def crossover(target, mutant):
        return [
            mutant[i] if random.random() < crossover_rate else target[i]
            for i in range(dim)
        ]

    population = [create_individual() for _ in range(population_size)]
    fitness_population = [func(individual) for individual in population]
    evals += population_size

    while evals < max_evals:
        for idx in range(population_size):
            mutant_vector = mutate(idx, population)
            trial_vector = crossover(population[idx], mutant_vector)
            trial_fitness = func(trial_vector)
            evals += 1

            if trial_fitness < fitness_population[idx]:
                population[idx] = trial_vector
                fitness_population[idx] = trial_fitness

    best_fitness = min(fitness_population)
    return best_fitness

#
#To improve performance, I've adjusted the population size, mutation factor, and crossover rate to potentially provide better exploration and exploitation balance. This approach can result in better convergence and more significant differences in benchmark tests.
