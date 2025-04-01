import random

def run(func, dim, bounds, max_evals):
    # Initialize
    best = float('inf')
    best_solution = [0.0] * dim
    eval_count = 0

    # Generate initial population
    population_size = 20
    population = []
    for _ in range(population_size):
        individual = [random.uniform(low, high) for low, high in bounds]
        population.append(individual)

    # Evaluate initial population
    population_fitness = []
    for individual in population:
        fitness = func(individual)
        eval_count += 1
        population_fitness.append(fitness)
        if fitness < best:
            best = fitness
            best_solution = individual

    # Algorithm main loop
    while eval_count < max_evals:
        new_population = []

        # Create new solutions through crossover and mutation
        for _ in range(population_size):
            # Select two parents from the population
            parent1, parent2 = random.sample(population, 2)

            # Crossover
            crossover_point = random.randint(1, dim - 1)
            offspring = parent1[:crossover_point] + parent2[crossover_point:]

            # Mutation
            mutation_probability = 0.1
            if random.random() < mutation_probability:
                mutation_index = random.randint(0, dim - 1)
                # Mutation within bounds
                offspring[mutation_index] = random.uniform(bounds[mutation_index][0], bounds[mutation_index][1])

            # Add the offspring to the new population
            new_population.append(offspring)

        # Evaluate new population
        for individual in new_population:
            fitness = func(individual)
            eval_count += 1
            if fitness < best:
                best = fitness
                best_solution = individual

            # Update population
            worst_fitness_index = population_fitness.index(max(population_fitness))
            if fitness < population_fitness[worst_fitness_index]:
                population[worst_fitness_index] = individual
                population_fitness[worst_fitness_index] = fitness

            if eval_count >= max_evals:
                break

    return best
