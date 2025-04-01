import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    population_size = 50  # Further increased population size for diversity
    best = float('inf')
    eval_count = 0

    # Generate initial population with random solutions
    population = []
    for _ in range(population_size):
        individual = [random.uniform(low, high) for low, high in bounds]
        population.append(individual)

    # Evaluate initial population
    fitness_values = [func(individual) for individual in population]
    eval_count += population_size
    best_index = min(range(population_size), key=lambda i: fitness_values[i])
    best = fitness_values[best_index]
    best_individual = population[best_index]

    # Main loop for enhanced Differential Evolution algorithm
    F_base = 0.9  # Base differential weight
    CR = 0.9  # Crossover probability

    while eval_count < max_evals:
        for i in range(population_size):
            # Adapt F dynamically based on fitness landscape
            F = F_base * (1 - (best / (best + fitness_values[i])))

            # Mutation strategy includes global best for exploitation
            indices = list(range(0, i)) + list(range(i + 1, population_size))
            a, b, = random.sample(indices, 2)
            target = population[i]
            donor = [
                best_individual[j] + F * (population[a][j] - population[b][j])
                for j in range(dim)
            ]

            # Crossover
            trial = [
                donor[j] if random.random() < CR else target[j]
                for j in range(dim)
            ]

            # Ensure trial is within bounds
            trial = [
                max(min(trial[j], bounds[j][1]), bounds[j][0])
                for j in range(dim)
            ]

            # Selection
            fitness = func(trial)
            eval_count += 1
            if fitness < fitness_values[i]:
                population[i] = trial
                fitness_values[i] = fitness
                if fitness < best:
                    best = fitness
                    best_individual = trial

            if eval_count >= max_evals:
                break

        # Apply elitism more assertively
        worst_index = max(range(population_size), key=lambda i: fitness_values[i])
        if fitness_values[worst_index] > best:
            population[worst_index] = best_individual
            fitness_values[worst_index] = best

    # Return the fitness of the best found solution
    return best
