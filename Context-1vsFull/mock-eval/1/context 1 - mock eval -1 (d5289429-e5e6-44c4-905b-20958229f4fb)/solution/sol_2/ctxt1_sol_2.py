def run(func, dim, bounds, max_evals):
    import random

    # Define helper function for boundary checking and generation of random solutions
    def clip(params, bounds):
        return [max(min(params[i], bounds[i][1]), bounds[i][0]) for i in range(dim)]

    # Initialize parameters
    population_size = 50
    mutation_strength = 0.1
    leaders_ratio = 0.2

    # Calculate number of leaders
    num_leaders = max(1, int(population_size * leaders_ratio))

    # Initialize population
    population = [clip([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)], bounds) for _ in range(population_size)]

    # Evaluate initial population
    population_fitness = [func(ind) for ind in population]

    # Initialize best solution
    best_fitness = float('inf')
    best_solution = None

    evals = population_size

    # Main loop
    while evals < max_evals:
        # Sort population according to fitness
        sorted_indices = sorted(range(population_size), key=lambda i: population_fitness[i])
        
        # Select leaders
        leaders = [population[i] for i in sorted_indices[:num_leaders]]

        # Generate new population around leaders using diversification and mutation
        new_population = []
        while len(new_population) < population_size:
            for leader in leaders:
                mutated_individual = clip(
                    [leader[i] + random.gauss(0, mutation_strength) * (bounds[i][1] - bounds[i][0]) 
                     for i in range(dim)], bounds
                )
                new_population.append(mutated_individual)
                if len(new_population) >= population_size:
                    break

        # Evaluate new population
        population = new_population
        population_fitness = [func(ind) for ind in population]
        evals += len(population_fitness)

        # Update best solution
        for i, fitness in enumerate(population_fitness):
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[i]

    return best_fitness
