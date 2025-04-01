import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    population_size = 30  # Increased population size
    best = float('inf')
    eval_count = 0
    
    # Generate initial population
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
    
    # Main loop for an improved Differential Evolution-like algorithm
    F = 0.8  # Modified Differential weight
    CR = 0.9  # Increased Crossover probability

    while eval_count < max_evals:
        for i in range(population_size):
            # Mutation
            indices = list(range(0, i)) + list(range(i + 1, population_size))
            a, b, c = random.sample(indices, 3)
            target = population[i]
            donor = [
                population[a][j] + F * (population[b][j] - population[c][j])
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
        
        # Apply elitism: Keep the best solution found so far
        worst_index = max(range(population_size), key=lambda i: fitness_values[i])
        if fitness_values[worst_index] > best:
            population[worst_index] = best_individual
            fitness_values[worst_index] = best

    # return fitness of the best found solution
    return best
