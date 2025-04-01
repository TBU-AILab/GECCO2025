import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        """Ensure x is within the given bounds."""
        return [max(lower[i], min(x[i], upper[i])) for i in range(len(x))]

    def crossover(parent1, parent2):
        """Perform a blend crossover."""
        alpha = random.uniform(0.2, 0.8)
        return [(alpha * p1 + (1 - alpha) * p2) for p1, p2 in zip(parent1, parent2)]

    def mutate(x, mutation_strength):
        """Mutate the solution with a Gaussian perturbation."""
        return [xi + mutation_strength * random.gauss(0, 1) for xi in x]

    # Initialize variables
    population_size = 50
    initial_mutation_strength = 0.2
    evals = 0

    # Generate initial population
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals += len(population)

    # Main optimization loop
    while evals < max_evals:
        new_population = []

        # Dynamic mutation strength adjustment based on diversity
        population_avg = sum(fitnesses) / population_size
        population_std = (sum((f - population_avg) ** 2 for f in fitnesses) / population_size) ** 0.5
        mutation_strength = initial_mutation_strength * (1 if population_std == 0 else (population_std / (population_avg + 1e-8)))

        # Create a new generation
        for _ in range(population_size // 2):
            # Tournament selection
            tournament_size = 5
            parents_indices = random.sample(range(population_size), tournament_size)
            parents = [population[i] for i in parents_indices]
            parent_fitnesses = [fitnesses[i] for i in parents_indices]

            idx1 = min(range(tournament_size), key=lambda i: parent_fitnesses[i])
            idx2 = min((i for i in range(tournament_size) if i != idx1), key=lambda i: parent_fitnesses[i])

            parent1 = parents[idx1]
            parent2 = parents[idx2]

            # Crossover and mutation
            offspring1 = mutate(clip(crossover(parent1, parent2), [b[0] for b in bounds], [b[1] for b in bounds]), mutation_strength)
            offspring2 = mutate(clip(crossover(parent2, parent1), [b[0] for b in bounds], [b[1] for b in bounds]), mutation_strength)

            # Evaluate offsprings
            fitness1 = func(offspring1)
            fitness2 = func(offspring2)
            evals += 2

            # Add to new population
            new_population.extend([(offspring1, fitness1), (offspring2, fitness2)])

        # Form the new combined population and sort based on fitness
        combined_population = new_population + list(zip(population, fitnesses))
        combined_population.sort(key=lambda x: x[1])
        population, fitnesses = zip(*combined_population[:population_size])

    # Return the best found solution's fitness
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
