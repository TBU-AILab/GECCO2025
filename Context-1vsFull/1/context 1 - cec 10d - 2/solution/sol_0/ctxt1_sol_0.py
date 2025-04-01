import random

def run(func, dim, bounds, max_evals):
    # Initialize the population
    population_size = 20
    mutation_factor = 0.8
    crossover_rate = 0.7
    evals = 0

    def create_individual():
        return [random.uniform(low, high) for low, high in bounds]

    def mutate(candidate, population):
        # Ensure unique indices for mutation
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

    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    fitness_population = [func(individual) for individual in population]
    evals += population_size

    # Main loop
    while evals < max_evals:
        for idx in range(population_size):
            # Mutation
            mutant_vector = mutate(idx, population)
            # Crossover
            trial_vector = crossover(population[idx], mutant_vector)
            # Evaluate the trial vector
            trial_fitness = func(trial_vector)
            evals += 1

            # Selection
            if trial_fitness < fitness_population[idx]:
                population[idx] = trial_vector
                fitness_population[idx] = trial_fitness

        # Find the best solution
        best_fitness = min(fitness_population)
        best_index = fitness_population.index(best_fitness)

    return fitness_population[best_index]

