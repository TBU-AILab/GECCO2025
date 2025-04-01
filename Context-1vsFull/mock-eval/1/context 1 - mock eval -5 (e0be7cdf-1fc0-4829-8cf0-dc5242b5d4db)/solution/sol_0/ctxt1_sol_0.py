import random

def run(func, dim, bounds, max_evals):
    # Initialize variables
    best, evals, population_size = float('inf'), 0, 30
    population = [ [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]

    # Define a helper function to evaluate the fitness
    def evaluate(individual):
        nonlocal evals
        fitness = func(individual)
        evals += 1
        return fitness
    
    # Evaluate the initial population
    fitnesses = [evaluate(individual) for individual in population]

    while evals < max_evals:
        # Create new individuals through a combination of genetic operations
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(population, 2)
            # Crossover
            crossover_point = random.randint(1, dim - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            # Mutation
            if random.random() < 0.1:  # mutation rate
                mutate_index = random.randrange(dim)
                child[mutate_index] = random.uniform(bounds[mutate_index][0], bounds[mutate_index][1])
            new_population.append(child)
        
        # Evaluate the new population
        new_fitnesses = [evaluate(individual) for individual in new_population]

        # Select the best individuals for the next generation
        combined_population = population + new_population
        combined_fitnesses = fitnesses + new_fitnesses
        
        # Sort population based on fitness
        sorted_population = sorted(zip(combined_population, combined_fitnesses), key=lambda x: x[1])
        
        # Select the top solutions
        population, fitnesses = zip(*sorted_population[:population_size])

        # Update the best solution found
        if fitnesses[0] < best:
            best = fitnesses[0]

    return best
