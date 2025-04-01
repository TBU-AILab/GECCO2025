import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        """Ensures each element in x is within the specified bounds."""
        return [max(lower[i], min(x[i], upper[i])) for i in range(dim)]

    def crossover(parent1, parent2):
        """Performs uniform crossover."""
        return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

    def mutate(x, mutation_strength):
        """Applies Gaussian mutation."""
        return [xi + mutation_strength * random.gauss(0, 1) for xi in x]

    # Initialize algorithm parameters
    population_size = 50
    initial_mutation_strength = 0.5
    evals = 0

    # Create initial population
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals += population_size

    # Main optimization loop
    while evals < max_evals:
        new_population = []

        # Adjust mutation strength with respect to progress
        sorted_fitnesses = sorted(fitnesses)
        best_fitness = sorted_fitnesses[0]
        diversity = max(sorted_fitnesses) - best_fitness
        mutation_strength = initial_mutation_strength * (1 + diversity / (abs(best_fitness) + 1e-8))

        for _ in range(population_size // 2):
            # Tournament selection
            candidates = random.sample(range(population_size), 3)
            best_candidates = sorted(candidates, key=lambda idx: fitnesses[idx])
            parent1_idx = best_candidates[0]
            parent2_idx = best_candidates[1]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover and mutation
            offspring1 = mutate(crossover(parent1, parent2), mutation_strength)
            offspring2 = mutate(crossover(parent2, parent1), mutation_strength)

            # Clip offspring
            offspring1 = clip(offspring1, [b[0] for b in bounds], [b[1] for b in bounds])
            offspring2 = clip(offspring2, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate offspring
            fitness1 = func(offspring1)
            fitness2 = func(offspring2)
            evals += 2

            # Add offspring to the new population
            new_population.extend([(offspring1, fitness1), (offspring2, fitness2)])

        # Form the new combined population
        combined_population = new_population + list(zip(population, fitnesses))
        combined_population.sort(key=lambda x: x[1])
        population, fitnesses = zip(*combined_population[:population_size])

    # Return the best solution's fitness found
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
