import random

def run(func, dim, bounds, max_evals):
    best_solution = None
    best_fitness = float('inf')
    evals = 0

    # Initialize population
    population_size = 20
    population = [
        [random.uniform(low, high) for low, high in bounds]
        for _ in range(population_size)
    ]

    def evaluate_solution(solution):
        nonlocal evals
        evals += 1
        return func(solution)

    # Evaluate initial population
    fitness_values = [
        evaluate_solution(individual) for individual in population
    ]

    # Update global best
    min_fitness_index = fitness_values.index(min(fitness_values))
    if fitness_values[min_fitness_index] < best_fitness:
        best_fitness = fitness_values[min_fitness_index]
        best_solution = population[min_fitness_index]

    # Main loop
    while evals < max_evals:
        # Select parents for crossover
        parents = random.sample(population, 2)

        # Crossover
        crossover_point = random.randint(1, dim - 1)
        offspring = parents[0][:crossover_point] + parents[1][crossover_point:]

        # Mutation
        mutation_probability = 0.1
        for i in range(dim):
            if random.random() < mutation_probability:
                low, high = bounds[i]
                offspring[i] = random.uniform(low, high)

        # Evaluate offspring
        offspring_fitness = evaluate_solution(offspring)

        # Selection
        worst_idx = fitness_values.index(max(fitness_values))
        if offspring_fitness < fitness_values[worst_idx]:
            population[worst_idx] = offspring
            fitness_values[worst_idx] = offspring_fitness

            # Update global best
            if offspring_fitness < best_fitness:
                best_fitness = offspring_fitness
                best_solution = offspring

    return best_fitness
