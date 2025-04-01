import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        return [max(lower[i], min(x[i], upper[i])) for i in range(len(x))]

    def crossover(parent1, parent2):
        return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

    def mutate(x, mutation_rate):
        return [xi + mutation_rate * random.uniform(-1, 1) for xi in x]

    # Initialize algorithm variables
    population_size = 20
    mutation_rate = 0.05
    tournament_size = 3
    num_generations = max_evals // population_size

    # Initialize population
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals = len(population)

    # Main loop
    for _ in range(num_generations):
        new_population = []

        while len(new_population) < population_size and evals < max_evals:
            # Tournament selection
            tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
            parent1 = min(tournament, key=lambda x: x[1])[0]
            parent2 = min(random.sample(list(zip(population, fitnesses)), tournament_size), key=lambda x: x[1])[0]

            # Crossover
            offspring = crossover(parent1, parent2)

            # Mutation
            offspring = mutate(offspring, mutation_rate)

            # Ensure offspring is within bounds
            offspring = clip(offspring, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate offspring
            offspring_fitness = func(offspring)
            evals += 1

            # Add offspring to new population
            new_population.append((offspring, offspring_fitness))

        # Form the new population for the next generation
        population, fitnesses = zip(*sorted(new_population + list(zip(population, fitnesses)), key=lambda x: x[1])[:population_size])

    # Return the best found solution's fitness
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
