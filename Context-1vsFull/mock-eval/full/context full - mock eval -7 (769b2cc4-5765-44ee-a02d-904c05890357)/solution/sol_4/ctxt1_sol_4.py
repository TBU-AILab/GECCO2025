import random

def run(func, dim, bounds, max_evals):
    # Differential Evolution Algorithm Parameters
    population_size = 20
    mutation_factor = 0.8
    crossover_rate = 0.9

    def generate_candidate(bounds):
        """Generate a candidate solution within given bounds."""
        return [random.uniform(low, high) for low, high in bounds]

    # Initialize population with random candidates
    population = [generate_candidate(bounds) for _ in range(population_size)]
    fitness = [func(individual) for individual in population]
    evals = population_size

    # Identify initial best solution
    best_idx = fitness.index(min(fitness))
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    while evals < max_evals:
        for i in range(population_size):
            # Mutation: Create a mutant vector
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = random.sample(indices, 3)

            mutant = [
                min(max(population[a][j] + mutation_factor * (population[b][j] - population[c][j]), bounds[j][0]), bounds[j][1])
                for j in range(dim)
            ]

            # Crossover: Create trial vector from mutant and current candidate
            trial = [
                mutant[j] if random.random() < crossover_rate else population[i][j]
                for j in range(dim)
            ]

            # Evaluate trial solution
            trial_fitness = func(trial)
            evals += 1

            # Replacement: If the trial is better, replace the old solution
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                # Update the best known solution
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            if evals >= max_evals:
                break    

    return best_fitness
