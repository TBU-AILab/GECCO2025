import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    population_size = 20
    mutation_rate = 0.1
    crossover_rate = 0.7

    # Initialize population
    population = [
        [random.uniform(low, high) for low, high in bounds]
        for _ in range(population_size)
    ]
    fitness = [func(individual) for individual in population]
    eval_count = population_size  # Account for initial fitness evaluations

    # Find the best initial solution
    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_fitness = fitness[best_idx]

    while eval_count < max_evals:
        # Selection (Tournament Selection)
        selected = []
        for _ in range(population_size):
            i1, i2 = random.sample(range(population_size), 2)
            selected.append(population[i1] if fitness[i1] < fitness[i2] else population[i2])

        # Crossover
        children = []
        for _ in range(0, population_size, 2):
            parent1, parent2 = random.sample(selected, 2)
            if random.random() < crossover_rate:
                cut = random.randint(1, dim - 1)
                child1 = parent1[:cut] + parent2[cut:]
                child2 = parent2[:cut] + parent1[cut:]
            else:
                child1, child2 = parent1, parent2
            children.extend([child1, child2])

        # Mutation
        for child in children:
            if random.random() < mutation_rate:
                index = random.randint(0, dim - 1)
                low, high = bounds[index]
                child[index] = random.uniform(low, high)

        # Evaluate children
        new_fitness = [func(child) for child in children]
        eval_count += population_size

        # Survivor selection: replace worst individuals in population
        for i in range(population_size):
            if new_fitness[i] < fitness[i]:
                population[i], fitness[i] = children[i], new_fitness[i]

        # Update the best solution found
        current_best_idx = min(range(population_size), key=lambda i: fitness[i])
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
    
    # Return fitness of the best found solution
    return best_fitness
#
