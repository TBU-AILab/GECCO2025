import random

def run(func, dim, bounds, max_evals):
    # Initialization
    population_size = 20
    mutation_factor = 0.8
    crossover_rate = 0.9

    def generate_candidate(bounds):
        return [random.uniform(low, high) for low, high in bounds]

    population = [generate_candidate(bounds) for _ in range(population_size)]
    fitness = [func(ind) for ind in population]
    evals = population_size

    best_idx = fitness.index(min(fitness))
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    while evals < max_evals:
        for i in range(population_size):
            # Mutation
            [a, b, c] = random.sample([x for x in range(population_size) if x != i], 3)
            mutant = [
                population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                for j in range(dim)
            ]
            
            # Ensure mutant is within bounds
            mutant = [
                min(max(mutant[j], bounds[j][0]), bounds[j][1])
                for j in range(dim)
            ]

            # Crossover
            trial = []
            for j in range(dim):
                if random.random() < crossover_rate or j == dim - 1:
                    trial.append(mutant[j])
                else:
                    trial.append(population[i][j])

            # Selection
            trial_fitness = func(trial)
            evals += 1

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                # Check if this individual is the new best
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            if evals >= max_evals:
                break

    return best_fitness
