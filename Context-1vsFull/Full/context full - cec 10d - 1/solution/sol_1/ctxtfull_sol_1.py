import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    evals = 0
    best = float('inf')
    pop_size = 20
    mutation_rate = 0.05
    crossover_rate = 0.7

    # Randomly initialize a population within the bounds
    population = [[random.uniform(low, high) for low, high in bounds] for _ in range(pop_size)]
    fitness = [float('inf')] * pop_size

    for i in range(pop_size):
        fitness[i] = func(population[i])
        evals += 1
        if fitness[i] < best:
            best = fitness[i]

    while evals < max_evals:
        # Select parents using tournament selection
        tournament_size = 3
        parent1_index = min(random.sample(range(pop_size), tournament_size), key=lambda i: fitness[i])
        parent2_index = min(random.sample(range(pop_size), tournament_size), key=lambda i: fitness[i])
        
        parent1, parent2 = population[parent1_index], population[parent2_index]

        # Crossover to create a new child
        child = parent1[:]
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, dim - 1)  # Ensure at least one gene is exchanged
            child = parent1[:crossover_point] + parent2[crossover_point:]

        # Mutation to introduce variability
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
