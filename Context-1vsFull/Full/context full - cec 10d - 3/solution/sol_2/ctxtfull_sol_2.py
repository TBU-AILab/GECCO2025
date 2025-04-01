import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    evals = 0

    # Initialize parameters
    population_size = 100  # Enhanced population for better exploration
    archive = []  # Introduce an archive to store the best solutions
    archive_size = 20

    # Initialize the population
    population = [([random.uniform(low, high) for low, high in bounds], float('inf')) for _ in range(population_size)]

    # Evaluate initial population
    for i in range(population_size):
        position, _ = population[i]
        fitness = func(position)
        population[i] = (position, fitness)
        evals += 1
        # Add the best to archive
        if len(archive) < archive_size:
            archive.append((position, fitness))
        else:
            archive.sort(key=lambda x: x[1])
            if fitness < archive[-1][1]:
                archive[-1] = (position, fitness)
        if fitness < best:
            best = fitness

    mutation_prob = 0.15  # Mutation probability
    crossover_rate = 0.8  # Crossover rate

    # Main loop of the algorithm
    while evals < max_evals:
        new_population = []

        for _ in range(population_size):
            if evals >= max_evals:
                break
            
            # Select parents using tournament selection
            tournament_size = 4
            parents = random.sample(population + archive, tournament_size)
            parents.sort(key=lambda x: x[1])
            parent1, parent2 = parents[0][0], parents[1][0]

            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, dim - 1)
                child_position = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child_position = parent1 if random.random() > 0.5 else parent2

            # Mutation
            if random.random() < mutation_prob:
                mutate_index = random.randint(0, dim - 1)
                low, high = bounds[mutate_index]
                child_position[mutate_index] = random.uniform(low, high)

            # Evaluate new child
            fitness = func(child_position)
            evals += 1
            new_population.append((child_position, fitness))

            # Update the archive and best solution
            if fitness < best:
                best = fitness
                if len(archive) < archive_size:
                    archive.append((child_position, fitness))
                else:
                    archive.sort(key=lambda x: x[1])
                    if fitness < archive[-1][1]:
                        archive[-1] = (child_position, fitness)

        # Update the population with the best solutions
        population = sorted(population + new_population, key=lambda x: x[1])[:population_size]

    return best
