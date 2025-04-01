import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    evals = 0
    best = float('inf')
    pop_size = 100
    mutation_rate = 0.2
    crossover_rate = 0.9
    elite_ratio = 0.2

    def evaluate_population(population):
        nonlocal evals, best
        fitness = []
        for individual in population:
            f = func(individual)
            evals += 1
            fitness.append(f)
            if f < best:
                best = f
        return fitness

    # Randomly initialize a population within the bounds
    population = [[random.uniform(low, high) for low, high in bounds] for _ in range(pop_size)]
    fitness = evaluate_population(population)

    while evals < max_evals:
        # Sort population based on fitness
        sorted_indices = sorted(range(pop_size), key=lambda i: fitness[i])
        population = [population[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]

        # Elitism: Retain the top elite_ratio% of the population
        elite_count = int(pop_size * elite_ratio)
        new_population = population[:elite_count]

        # Generate new solutions using crossover and mutation
        while len(new_population) < pop_size:
            # Select parents using tournament selection
            tournament_size = 5
            parent_indices = random.sample(range(elite_count), 2)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]

            # Crossover to create a new child
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, dim - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent1[:]  # No crossover, just clone parent1

            # Mutation to introduce variability
            for j in range(dim):
                if random.random() < mutation_rate:
                    child[j] = random.uniform(bounds[j][0], bounds[j][1])

            new_population.append(child)

        # Evaluate the new population
        fitness = evaluate_population(new_population)

    # Return the fitness of the best-found solution
    return best
