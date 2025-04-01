import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    evals = 0
    best = float('inf')
    pop_size = 10
    mutation_rate = 0.1

    # Randomly initialize a population within the bounds
    population = [[random.uniform(low, high) for low, high in bounds] for _ in range(pop_size)]
    fitness = [float('inf')] * pop_size

    for i in range(pop_size):
        fitness[i] = func(population[i])
        evals += 1
        if fitness[i] < best:
            best = fitness[i]

    while evals < max_evals:
        # Select parent using tournament selection
        parent_index = min(range(pop_size), key=lambda i: fitness[i])
        parent = population[parent_index]

        # Mutation to create a new solution
        child = parent[:]  # Copy the parent's solution
        for j in range(dim):
            if random.random() < mutation_rate:
                child[j] = random.uniform(bounds[j][0], bounds[j][1])

        # Evaluate the new child
        child_fitness = func(child)
        evals += 1
        if child_fitness < best:
            best = child_fitness

        # Replace the worst individual in the population if the child is better
        worst_index = max(range(pop_size), key=lambda i: fitness[i])
        if child_fitness < fitness[worst_index]:
            population[worst_index] = child
            fitness[worst_index] = child_fitness

    # Return the fitness of the best-found solution
    return best
