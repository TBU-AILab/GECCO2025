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
    num_leaders = int(population_size * leaders_ratio)

    # Initialize population
    population = [clip([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)], bounds) for _ in range(population_size)]

    # Evaluate initial population
    population_fitness = [func(ind) for ind in population]

    # Initialize best solution
    best_fitness = min(population_fitness)
    best_index = population_fitness.index(best_fitness)
    best_solution = population[best_index]

    evals = population_size

    while evals < max_evals:
        # Sort population according to fitness
        sorted_indices = sorted(range(population_size), key=lambda i: population_fitness[i])
        
        # Select leaders
        leaders = [population[i] for i in sorted_indices[:num_leaders]]

        # Generate new population around leaders using a basic mutation
        new_population = []
        for leader in leaders:
            for _ in range(population_size // num_leaders):
                mutated_individual = clip(
                    [leader[i] + random.gauss(0, mutation_strength) * (bounds[i][1] - bounds[i][0]) 
                     for i in range(dim)], bounds
                )
                new_population.append(mutated_individual)

        # Evaluate new population
        population = new_population
        population_fitness = [func(ind) for ind in population]
        evals += len(population_fitness)

        # Update best solution
        current_best_fitness = min(population_fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_index = population_fitness.index(current_best_fitness)
            best_solution = population[best_index]

    return best_fitness
