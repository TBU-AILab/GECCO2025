import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        return [max(lower[i], min(x[i], upper[i])) for i in range(len(x))]

    def crossover(parent1, parent2):
        return [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]

    def mutate(x, mutation_strength):
        return [xi + mutation_strength * random.gauss(0, 1) for xi in x]

    # Initialize algorithm variables
    population_size = 50
    initial_mutation_strength = 0.5
    evals = 0

    # Initialize population
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals += len(population)

    # Main loop
    while evals < max_evals:
        new_population = []

        # Dynamic mutation strength
        sorted_indices = sorted(range(population_size), key=lambda i: fitnesses[i])
        best_fitness = fitnesses[sorted_indices[0]]
        mutation_strength = initial_mutation_strength * (1 - best_fitness/max_evals) if max_evals > 0 else initial_mutation_strength

        # Create a new generation
        for _ in range(population_size // 2):
            # Tournament selection
            parent_indices = random.sample(range(population_size), 3)
            parent1 = min(parent_indices, key=lambda idx: fitnesses[idx])
            parent2 = min(random.sample(range(population_size), 3), key=lambda idx: fitnesses[idx])

            # Crossover
            offspring1 = crossover(population[parent1], population[parent2])
            offspring2 = crossover(population[parent2], population[parent1])

            # Mutation
            offspring1 = mutate(offspring1, mutation_strength)
            offspring2 = mutate(offspring2, mutation_strength)

            # Clip to bounds
            offspring1 = clip(offspring1, [b[0] for b in bounds], [b[1] for b in bounds])
            offspring2 = clip(offspring2, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate
            fitness1 = func(offspring1)
            fitness2 = func(offspring2)
            evals += 2

            # Add to new population
            new_population.extend([(offspring1, fitness1), (offspring2, fitness2)])

        # Form the new population
        combined_population = new_population + list(zip(population, fitnesses))
        combined_population.sort(key=lambda x: x[1])
        population, fitnesses = zip(*combined_population[:population_size])

    # Return the best found solution's fitness
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
