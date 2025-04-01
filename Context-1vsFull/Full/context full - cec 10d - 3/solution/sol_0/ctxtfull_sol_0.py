import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    best_position = None
    evals = 0

    # Initialize the population
    population_size = 20
    population = [([random.uniform(low, high) for low, high in bounds], float('inf')) for _ in range(population_size)]

    # Evaluate initial population
    for i in range(population_size):
        position, _ = population[i]
        fitness = func(position)
        population[i] = (position, fitness)
        evals += 1
        if fitness < best:
            best = fitness
            best_position = position

    # Main loop of the algorithm
    while evals < max_evals:
        new_population = []

        for _ in range(population_size):
            # Mutation and Crossover
            if evals >= max_evals:
                break
            
            parent1, parent2 = random.sample(population, 2)

            # Crossover
            crossover_point = random.randint(1, dim - 1)
            child_position = parent1[0][:crossover_point] + parent2[0][crossover_point:]
            
            # Mutation
            mutation_prob = 0.1
            if random.random() < mutation_prob:
                mutate_index = random.randint(0, dim - 1)
                low, high = bounds[mutate_index]
                child_position[mutate_index] = random.uniform(low, high)

            # Evaluate new child
            fitness = func(child_position)
            evals += 1
            new_population.append((child_position, fitness))

            if fitness < best:
                best = fitness
                best_position = child_position

        # Replace the old population with the new one
        population = sorted(population + new_population, key=lambda x: x[1])[:population_size]

    return best
#
