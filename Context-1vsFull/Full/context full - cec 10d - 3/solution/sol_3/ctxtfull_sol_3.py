import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    evals = 0

    # Initialize parameters
    population_size = 120  # Further increased for better exploration
    archive = []  # Archive to store the elite solutions
    archive_size = 30

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

    mutation_prob_dynamic = 0.25  # Dynamic mutation
    crossover_rate = 0.85  # Elevated crossover rate

    # Main loop of the algorithm
    while evals < max_evals:
        new_population = []

        for _ in range(population_size):
            if evals >= max_evals:
                break
            
            # Select parents using a blend of tournament and random from archive
            tournament_size = 5
            parents = random.sample(population, tournament_size) + random.sample(archive, min(2, len(archive)))
            parents.sort(key=lambda x: x[1])
            parent1, parent2 = parents[0][0], parents[1][0]

            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, dim - 1)
                child_position = [
                    parent1[i] if random.random() < 0.5 else parent2[i]
                    for i in range(dim)
                ]
            else:
                child_position = parent1 if random.random() > 0.5 else parent2

            # Introduce adaptive mutation based on diversity
            mutation_prob = mutation_prob_dynamic * (1 - (evals / max_evals))
            
            if random.random() < mutation_prob:
                mutate_index = random.randint(0, dim - 1)
                low, high = bounds[mutate_index]
                child_position[mutate_index] = random.uniform(low, high)

            # Evaluate new child
            fitness = func(child_position)
            evals += 1
            new_population.append((child_position, fitness))

            # Update archive and best solution
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
