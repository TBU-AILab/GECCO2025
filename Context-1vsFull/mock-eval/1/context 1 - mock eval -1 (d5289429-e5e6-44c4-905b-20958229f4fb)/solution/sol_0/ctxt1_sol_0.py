def run(func, dim, bounds, max_evals):
    
    # Initialize parameters
    import random

    # Define helper function for boundary checking and generation of random solutions
    def clip(params, bounds):
        return [max(min(params[i], bounds[i][1]), bounds[i][0]) for i in range(dim)]
        
    # Initialize a population size
    population_size = 50
    leaders_ratio = 0.2
    
    # Calculate number of leaders
    num_leaders = int(population_size * leaders_ratio)
    
    # Initialize population
    population = [clip([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)], bounds) for _ in range(population_size)]
    
    # Evaluate initial population
    population_fitness = [func(ind) for ind in population]
  
    best = min(population_fitness)
    evals = population_size
    
    while evals < max_evals:
        # Sort population according to fitness
        sorted_population = [x for _, x in sorted(zip(population_fitness, population))]
        
        # Select leaders
        leaders = sorted_population[:num_leaders]
        
        # Generate new population around leaders using a basic mutation
        new_population = []
        for leader in leaders:
            for _ in range(population_size // num_leaders):
                mutated_individual = clip([leader[i] + random.uniform(-0.1, 0.1) * (bounds[i][1] - bounds[i][0]) for i in range(dim)], bounds)
                new_population.append(mutated_individual)
        
        # Clip the unwanted offsprings if any
        if len(new_population) > population_size:
            new_population = new_population[:population_size]
        
        # Evaluate new population
        population = new_population
        population_fitness = [func(ind) for ind in population]
        evals += len(population_fitness)
        
        # Update best
        new_best = min(population_fitness)
        best = min(best, new_best)
    
    return best
