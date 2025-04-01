import random

def run(func, dim, bounds, max_evals):
    # Initialize variables for tracking the best solution
    best_solution = None
    best_fitness = float('inf')
    evals = 0
    
    # Define parameters for Differential Evolution
    F = 0.8  # Mutation factor, increased for exploration
    CR = 0.7  # Crossover probability, reduced for diversity
    population_size = 15 * dim  # Population size, increased for better coverage

    # Initialize population with random solutions within the bounds
    population = [
        [random.uniform(low, high) for low, high in bounds]
        for _ in range(population_size)
    ]
    
    # Evaluate initial population
    fitnesses = []
    for individual in population:
        fitness = func(individual)
        evals += 1
        fitnesses.append(fitness)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = individual

    while evals < max_evals:
        for i in range(population_size):
            # Ensure the algorithm doesn't exceed the maximum allowed evaluations
            if evals >= max_evals:
                break

            # Mutation: Select three unique indices a, b, c different from i
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)
            x = population[i]
            y = [
                population[a][j] + F * (population[b][j] - population[c][j])
                for j in range(dim)
            ]

            # Ensure the mutated vector is within the bounds
            y = [max(min(y[j], bounds[j][1]), bounds[j][0]) for j in range(dim)]

            # Crossover: Generate a trial solution
            trial = [
                y[j] if random.random() < CR else x[j]
                for j in range(dim)
            ]

            # Evaluate the trial vector
            trial_fitness = func(trial)
            evals += 1

            # Adaptive strategy: Introduce elitism for diversity
            if trial_fitness < fitnesses[i] or random.random() < 0.1:
                population[i] = trial
                fitnesses[i] = trial_fitness

                # Update the best solution found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

    # Return fitness of the best-found solution
    return best_fitness
