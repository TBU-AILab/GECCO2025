import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        """Ensure x is within the provided bounds."""
        return [max(lower[i], min(x[i], upper[i])) for i in range(dim)]

    def crossover(parent1, parent2):
        """Perform arithmetic crossover."""
        return [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]

    def mutate(x, mutation_strength):
        """Apply Gaussian mutation."""
        return [xi + mutation_strength * random.gauss(0, 1) for xi in x]

    # Initialize parameters
    population_size = 50
    mutation_strength = 0.1
    evals = 0

    # Initialize population and evaluate fitnesses
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals += len(population)

    # Main optimization loop
    while evals < max_evals:
        new_population = []

        # Adjust mutation strength based on population diversity
        fitness_variance = sum((f - sum(fitnesses) / population_size) ** 2 for f in fitnesses) / population_size
        mutation_strength = 0.1 + (0.5 - 0.1) * min(1, fitness_variance)

        for _ in range(population_size // 2):
            # Select parents using tournament selection
            tournament_size = 5
            indices = random.sample(range(population_size), tournament_size)
            fitness_values = [(i, fitnesses[i]) for i in indices]
            parent1_idx = min(fitness_values, key=lambda x: x[1])[0]

            indices.remove(parent1_idx)
            parent2_idx = min((i for i in indices), key=lambda i: fitnesses[i])

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Create offspring using crossover and mutation
            offspring1 = mutate(crossover(parent1, parent2), mutation_strength)
            offspring2 = mutate(crossover(parent2, parent1), mutation_strength)

            # Clip the offspring if they go out of bounds
            offspring1 = clip(offspring1, [b[0] for b in bounds], [b[1] for b in bounds])
            offspring2 = clip(offspring2, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate offspring
            fitness1 = func(offspring1)
            fitness2 = func(offspring2)
            evals += 2

            # Add offspring to the new population
            new_population.extend([(offspring1, fitness1), (offspring2, fitness2)])

        # Replace the old population with the new one, keeping the best individuals
        combined_population = new_population + list(zip(population, fitnesses))
        combined_population.sort(key=lambda x: x[1])
        population, fitnesses = zip(*combined_population[:population_size])

    # Return the fitness of the best found solution
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
