import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    evals = 0

    # Initialize parameters
    population_size = 150  # Further increase for more exploration
    archive = []  # Archive for elite solutions
    archive_size = 40

    # Initialize initial population
    population = [([random.uniform(low, high) for low, high in bounds], float('inf')) for _ in range(population_size)]

    # Evaluate initial population
    for i in range(population_size):
        position, _ = population[i]
        fitness = func(position)
        population[i] = (position, fitness)
        evals += 1
        if len(archive) < archive_size:
            archive.append((position, fitness))
        else:
            archive.sort(key=lambda x: x[1])
            if fitness < archive[-1][1]:
                archive[-1] = (position, fitness)
        if fitness < best:
            best = fitness

    mutation_prob_dynamic = 0.3  # Adaptive mutation probability
    crossover_rate = 0.9  # High crossover rate

    # Main loop of the algorithm
    while evals < max_evals:
        new_population = []

        for _ in range(population_size):
            if evals >= max_evals:
                break
            
            # Select parents using tournament selection
            tournament_size = 6
            candidates = random.sample(population, tournament_size) + random.sample(archive, min(4, len(archive)))
            candidates.sort(key=lambda x: x[1])
            parent1, parent2 = candidates[0][0], candidates[1][0]

            # Crossover with adaptive crossover rate
            if random.random() < crossover_rate:
                child_position = [(parent1[i] + parent2[i]) / 2 for i in range(dim)]
                if random.random() < 0.5:
                    crossover_point = random.randint(1, dim - 1)
                    child_position = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child_position = parent1 if random.random() > 0.5 else parent2

            # Adaptive mutation to enhance diversity
            mutation_prob = mutation_prob_dynamic * (1 - (evals / max_evals))
            if random.random() < mutation_prob:
                for _ in range(random.randint(1, dim // 2)):  # Mutate up to half of the dimensions
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
