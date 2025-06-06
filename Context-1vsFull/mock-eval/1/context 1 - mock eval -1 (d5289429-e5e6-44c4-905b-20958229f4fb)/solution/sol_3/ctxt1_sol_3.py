def run(func, dim, bounds, max_evals):
    import random

    # Define helper function for boundary checking and generation of random solutions
    def clip(params, bounds):
        return [max(min(params[i], bounds[i][1]), bounds[i][0]) for i in range(dim)]

    # Initialize parameters
    population_size = 100  # Increased population size for better exploration
    mutation_strength = 0.1
    crossover_rate = 0.7   # Added crossover parameter
    elite_ratio = 0.1      # Changed from leaders_ratio for clarity
    num_elites = max(1, int(population_size * elite_ratio))

    # Initialize population
    population = [clip([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)], bounds) for _ in range(population_size)]

    # Evaluate initial population
    population_fitness = [func(ind) for ind in population]

    # Initialize evaluations counter
    evals = population_size

    # Initialize best solution
    best_fitness = float('inf')
    best_solution = None

    # Main loop
    while evals < max_evals:
        # Sort population according to fitness
        sorted_indices = sorted(range(population_size), key=lambda i: population_fitness[i])

        # Select elites
        elites = [population[i] for i in sorted_indices[:num_elites]]

        # Generate new population using crossover and mutation
        new_population = elites.copy()
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                # Perform crossover
                parent1, parent2 = random.sample(elites, 2)
                crossover_point = random.randint(1, dim - 1)
                child = clip(
                    parent1[:crossover_point] + parent2[crossover_point:], bounds
                )
            else:
                # Mutation only
                parent = random.choice(elites)
                child = clip(
                    [parent[i] + random.gauss(0, mutation_strength) * (bounds[i][1] - bounds[i][0]) for i in range(dim)],
                    bounds
                )
            new_population.append(child)

        # Evaluate new population
        population = new_population[:population_size]
        population_fitness = [func(ind) for ind in population]
        evals += len(population_fitness)

        # Update best solution
        for i, fitness in enumerate(population_fitness):
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[i]

    return best_fitness, best_solution
