import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        return [max(lower[i], min(x[i], upper[i])) for i in range(len(x))]

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, dim - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(x, mutation_strength):
        return [xi + mutation_strength * random.gauss(0, 1) for xi in x]

    # Initialize algorithm variables
    population_size = 40
    initial_mutation_strength = 0.1
    mutation_decay = 0.99
    tournament_size = 3
    evals = 0

    # Initialize population
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals += len(population)

    # Main loop with adaptive mutation handling
    while evals < max_evals:
        new_population = []

        # Adjust mutation strength based on progress
        mutation_strength = initial_mutation_strength * (mutation_decay ** (evals // population_size))

        # Create a new generation
        for _ in range(population_size // 2):
            # Tournament selection
            tournament_candidates = random.sample(range(population_size), tournament_size)
            parent1_index = min(tournament_candidates, key=lambda idx: fitnesses[idx])
            parent2_index = min(random.sample(range(population_size), tournament_size), key=lambda idx: fitnesses[idx])

            parent1 = population[parent1_index]
            parent2 = population[parent2_index]

            # Crossover
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)

            # Mutation
            offspring1 = mutate(offspring1, mutation_strength)
            offspring2 = mutate(offspring2, mutation_strength)

            # Clip offsprings to bounds
            offspring1 = clip(offspring1, [b[0] for b in bounds], [b[1] for b in bounds])
            offspring2 = clip(offspring2, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate offsprings
            fitness1 = func(offspring1)
            fitness2 = func(offspring2)
            evals += 2

            # Add offspring to new population
            new_population.extend([(offspring1, fitness1), (offspring2, fitness2)])

        # Form the new population for the next generation
        combined_population = new_population + list(zip(population, fitnesses))
        combined_population.sort(key=lambda x: x[1])
        population, fitnesses = zip(*combined_population[:population_size])

    # Return the best found solution's fitness
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
