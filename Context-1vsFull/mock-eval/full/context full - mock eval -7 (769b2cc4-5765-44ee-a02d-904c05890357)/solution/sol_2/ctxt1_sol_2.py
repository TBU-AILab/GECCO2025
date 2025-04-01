import random

def run(func, dim, bounds, max_evals):
    # Parameters
    population_size = 20
    mutation_factor = 0.8
    crossover_rate = 0.9

    def generate_candidate(bounds):
        return [random.uniform(low, high) for low, high in bounds]

    # Initialize population
    population = [generate_candidate(bounds) for _ in range(population_size)]
    fitness = [func(individual) for individual in population]
    evals = population_size

    # Track best solution
    best_idx = fitness.index(min(fitness))
    best_fitness = fitness[best_idx]

    while evals < max_evals:
        new_population = []
        for i in range(population_size):
            # Ensure diverse set of random indices for mutation
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)

            # Mutation
            mutant = [
                population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                for j in range(dim)
            ]
            
            # Ensure mutant is within bounds
            mutant = [min(max(mutant[j], bounds[j][0]), bounds[j][1]) for j in range(dim)]

            # Crossover
            trial = [
                mutant[j] if random.random() < crossover_rate or j == dim - 1 else population[i][j]
                for j in range(dim)
            ]

            # Selection
            trial_fitness = func(trial)
            evals += 1

            # Keep the better of the trial and current individual
            if trial_fitness < fitness[i]:
                new_population.append(trial)
                fitness[i] = trial_fitness

                # Update the best solution found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
            else:
                new_population.append(population[i])

            if evals >= max_evals:
                break    

        # Replace population with new population
        population = new_population

    return best_fitness
