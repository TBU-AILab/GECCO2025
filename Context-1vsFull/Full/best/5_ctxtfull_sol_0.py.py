import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    population_size = 20
    best = float('inf')
    eval_count = 0
    
    # Generate initial population
    population = []
    for _ in range(population_size):
        individual = [random.uniform(low, high) for low, high in bounds]
        population.append(individual)
    
    # Evaluate initial population
    fitness_values = []
    for individual in population:
        fitness = func(individual)
        eval_count += 1
        fitness_values.append(fitness)
        if fitness < best:
            best = fitness
    
    # Main loop for a simple Differential Evolution-like algorithm
    F = 0.7  # Differential weight
    CR = 0.5  # Crossover probability

    while eval_count < max_evals:
        for i in range(population_size):
            # Mutation and Crossover
            indices = list(range(0, i)) + list(range(i + 1, population_size))
            a, b, c = random.sample(indices, 3)
            target = population[i]
            donor = [
                population[a][j] + F * (population[b][j] - population[c][j])
                for j in range(dim)
            ]
            
            trial = [
                donor[j] if random.random() < CR else target[j]
                for j in range(dim)
            ]
            
            # Ensure trial is within bounds
            trial = [
                max(min(trial[j], bounds[j][1]), bounds[j][0])
                for j in range(dim)
            ]

            # Select the better solution
            fitness = func(trial)
            eval_count += 1
            if fitness < fitness_values[i]:
                population[i] = trial
                fitness_values[i] = fitness
                if fitness < best:
                    best = fitness
            
            if eval_count >= max_evals:
                break

    # return fitness of the best found solution
    return best
