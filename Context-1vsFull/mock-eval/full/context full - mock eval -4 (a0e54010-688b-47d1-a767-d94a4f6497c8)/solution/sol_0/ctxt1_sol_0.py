import random

def run(func, dim, bounds, max_evals):
    def clip(x, lower, upper):
        return [max(lower[i], min(x[i], upper[i])) for i in range(len(x))]

    def crossover(parent1, parent2):
        alpha = random.uniform(0, 1)
        return [(alpha * p1 + (1 - alpha) * p2) for p1, p2 in zip(parent1, parent2)]

    def mutate(x, mutation_rate, lower, upper):
        return [x[i] + mutation_rate * random.uniform(-1, 1) if random.random() < mutation_prob else x[i] for i in range(dim)]

    # Initialize population
    population_size = 10
    mutation_prob = 0.1
    mutation_rate = 0.1
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    fitnesses = [func(ind) for ind in population]
    evals = len(population)

    # Main loop
    while evals < max_evals:
        # Select two parents randomly
        parent1, parent2 = random.sample(population, 2)

        # Generate an offspring through crossover
        offspring = crossover(parent1, parent2)

        # Apply mutation
        offspring = mutate(offspring, mutation_rate, [b[0] for b in bounds], [b[1] for b in bounds])

        # Ensure offspring is within bounds
        offspring = clip(offspring, [b[0] for b in bounds], [b[1] for b in bounds])

        # Evaluate offspring
        offspring_fitness = func(offspring)
        evals += 1

        # Replace the worst individual if offspring is better
        worst_index = fitnesses.index(max(fitnesses))
        if offspring_fitness < fitnesses[worst_index]:
            population[worst_index] = offspring
            fitnesses[worst_index] = offspring_fitness

    # Find and return the best solution
    best_index = fitnesses.index(min(fitnesses))
    return fitnesses[best_index]
