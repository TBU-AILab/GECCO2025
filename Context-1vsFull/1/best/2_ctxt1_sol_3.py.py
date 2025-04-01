import random

def run(func, dim, bounds, max_evals):
    # Optimizer configuration
    population_size = 50  # Increased to allow better diversity
    mutation_factor = 0.9  # Increased for stronger mutations
    crossover_rate = 0.7   # Reduced to encourage more exploration
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
#In this version, I've adjusted the population size, mutation factor, and crossover rate to achieve a better balance between exploration and exploitation. An increased mutation factor may help the optimizer explore more diverse solutions, while a slightly reduced crossover rate might allow the algorithm to escape local optima by retaining some diversity within the population. This configuration could improve performance and help address the provided feedback effectively.
