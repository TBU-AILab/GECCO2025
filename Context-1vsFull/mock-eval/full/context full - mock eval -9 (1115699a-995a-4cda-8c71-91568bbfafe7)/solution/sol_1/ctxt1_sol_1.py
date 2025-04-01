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
        
        # Dynamic mutation probability
        mutation_probability = 0.2 * (1 - eval_count/max_evals)

        # Create new solutions through crossover and mutation
        for _ in range(population_size):
            # Select two parents using tournament selection
            tournament_size = 3
            tournament = random.sample(population, tournament_size)
            parent1 = min(tournament, key=lambda ind: population_fitness[population.index(ind)])
            parent2 = min(random.sample(population, tournament_size), key=lambda ind: population_fitness[population.index(ind)])

            # Crossover using uniform crossover
            offspring = [parent1[i] if random.random() > 0.5 else parent2[i] for i in range(dim)]

            # Mutation
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

            # Replace the worst individual in the current population with the new individual if it's better
            if fitness < max(population_fitness):
                worst_fitness_index = population_fitness.index(max(population_fitness))
                population[worst_fitness_index] = individual
                population_fitness[worst_fitness_index] = fitness

            if eval_count >= max_evals:
                break

    return best
