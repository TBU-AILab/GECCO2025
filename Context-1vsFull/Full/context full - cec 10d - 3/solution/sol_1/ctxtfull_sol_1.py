import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    best_position = None
    evals = 0

    # Initialize the population
    population_size = 50  # Increased for a more diverse exploration
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

    mutation_prob = 0.2  # Increased mutation probability for diversity
    crossover_rate = 0.7  # New crossover rate for combining traits

    # Main loop of the algorithm
    while evals < max_evals:
        new_population = []

        for _ in range(population_size):
            if evals >= max_evals:
                break
            
            # Select parents using tournament selection
            tournament_size = 3
            parents = random.sample(population, tournament_size)
            parents.sort(key=lambda x: x[1])
            parent1, parent2 = parents[0][0], parents[1][0]

            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, dim - 1)
                child_position = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child_position = random.choice([parent1, parent2])

            # Mutation
            if random.random() < mutation_prob:
                mutate_index = random.randint(0, dim - 1)
                low, high = bounds[mutate_index]
                child_position[mutate_index] = random.uniform(low, high)

            # Evaluate new child
            fitness = func(child_position)
            evals += 1
            new_population.append((child_position, fitness))

            # Update the best solution
            if fitness < best:
                best = fitness
                best_position = child_position

        # Update the population with the best solutions
        population = sorted(population + new_population, key=lambda x: x[1])[:population_size]

    return best
#
