import random

def run(func, dim, bounds, max_evals):
    # Initialize variables for tracking the best solution
    best_solution = None
    best_fitness = float('inf')  # Start with the worst possible fitness
    evals = 0

    # Define parameters for Differential Evolution
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover probability, increased for better exploration
    population_size = 20 * dim  # Adequate population size for diversity

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
            best_solution = individual[:]

    while evals < max_evals:
        for i in range(population_size):
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

            # Record and check if the trial vector becomes a new best
            if trial_fitness < fitnesses[i]:
                population[i] = trial
                fitnesses[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial[:]

            # Stop the process if max evaluations reached
            if evals >= max_evals:
                break

    # Return fitness of the best-found solution and the best solution itself
    return best_fitness, best_solution

# Example usage (assuming a problem definition is provided):
# def objective_function(x):
#     return sum([xi ** 2 for xi in x])
# dim = 10
# bounds = [(-5, 5) for _ in range(dim)]
# max_evals = 1000
# best_cost, best_solution = run(objective_function, dim, bounds, max_evals)
# print(f"Best cost: {best_cost}, Best solution: {best_solution}")
