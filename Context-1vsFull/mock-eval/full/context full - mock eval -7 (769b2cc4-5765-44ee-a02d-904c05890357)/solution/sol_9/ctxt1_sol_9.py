import random

def run(func, dim, bounds, max_evals):
    # Parameters for the DE-like algorithm
    population_size = 20
    mutation_factor = 0.8
    crossover_rate = 0.9

    def generate_candidate(bounds):
        """Generate a random candidate solution within the specified bounds."""
        return [random.uniform(low, high) for low, high in bounds]

    # Initialize population with random candidate solutions
    population = [generate_candidate(bounds) for _ in range(population_size)]
    fitness = [func(individual) for individual in population]
    evals = population_size

    # Track the best solution found
    best_idx = fitness.index(min(fitness))
    best_fitness = fitness[best_idx]

    while evals < max_evals:
        for i in range(population_size):
            # Ensure a diverse set of indices for mutation
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)

            # Mutation creates a donor vector
            donor = [
                population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                for j in range(dim)
            ]

            # Bound correction for donor
            donor = [min(max(donor[j], bounds[j][0]), bounds[j][1]) for j in range(dim)]

            # Crossover creates a trial vector
            trial = [
                donor[j] if random.random() < crossover_rate else population[i][j]
                for j in range(dim)
            ]

            # Evaluate trial vector
            trial_fitness = func(trial)
            evals += 1

            # Replacement logic: Keep the better individual
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                # Update the best solution if trial is superior
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

            # Early exit if maximum evaluations are reached
            if evals >= max_evals:
                return best_fitness

    return best_fitness
