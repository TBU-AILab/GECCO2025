import random

def run(func, dim, bounds, max_evals):
    # Differential Evolution (DE) inspired algorithm parameters
    population_size = 20
    mutation_factor = 0.8
    crossover_rate = 0.9

    def generate_candidate(bounds):
        """Generate a random candidate within the specified bounds."""
        return [random.uniform(low, high) for low, high in bounds]

    # Initialize population with random candidates
    population = [generate_candidate(bounds) for _ in range(population_size)]
    fitness = [func(individual) for individual in population]
    evals = population_size

    # Track the best solution found
    best_idx = fitness.index(min(fitness))
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    while evals < max_evals:
        new_population = []
        for i in range(population_size):
            # Mutation: Create a mutant vector
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)

            mutant = [
                population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                for j in range(dim)
            ]

            # Ensure mutant values are within bounds
            mutant = [
                min(max(mutant[j], bounds[j][0]), bounds[j][1]) for j in range(dim)
            ]

            # Crossover: Create a trial vector from mutant and current vector
            trial = [
                mutant[j] if random.random() < crossover_rate else population[i][j]
                for j in range(dim)
            ]

            # Evaluate the trial vector
            trial_fitness = func(trial)
            evals += 1

            # Keep the best solution between the current individual and the trial
            if trial_fitness < fitness[i]:
                new_population.append(trial)
                fitness[i] = trial_fitness

                # Update best solution if the trial is better
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
            else:
                new_population.append(population[i])

            # Exit if the maximum evaluations are reached
            if evals >= max_evals:
                break
        
        # Update the population
        population = new_population

    return best_fitness
