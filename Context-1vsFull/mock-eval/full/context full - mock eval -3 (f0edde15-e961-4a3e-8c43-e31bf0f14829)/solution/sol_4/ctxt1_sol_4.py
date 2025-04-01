#Certainly! Let’s enhance the solution by incorporating additional strategies for mutation and selection, ensuring clearer handling of data types and improving overall efficiency.
#
import random

def run(func, dim, bounds, max_evals):
    # Initialize parameters
    population_size = 20
    mutation_rate = 0.1
    crossover_rate = 0.7

    # Initialize the population with random solutions within the bounds
    population = [
        [random.uniform(low, high) for low, high in bounds]
        for _ in range(population_size)
    ]
    fitness = [func(individual) for individual in population]
    eval_count = population_size  # Initial evaluations count

    # Track the best solution found
    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_fitness = fitness[best_idx]

    while eval_count < max_evals:
        # Selection: Tournament Selection
        selected = []
        for _ in range(population_size):
            i1, i2 = random.sample(range(population_size), 2)
            winner = population[i1] if fitness[i1] < fitness[i2] else population[i2]
            selected.append(winner)

        # Crossover and Mutation to create offspring
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(selected, 2)
            
            # Uniform Crossover
            if random.random() < crossover_rate:
                child1, child2 = [], []
                for gene1, gene2 in zip(parent1, parent2):
                    if random.random() < 0.5:
                        child1.append(gene1)
                        child2.append(gene2)
                    else:
                        child1.append(gene2)
                        child2.append(gene1)
            else:
                child1, child2 = parent1[:], parent2[:]
            offspring.extend([child1, child2])

        # Mutation
        for child in offspring:
            if random.random() < mutation_rate:
                index = random.randint(0, dim - 1)
                low, high = bounds[index]
                child[index] = random.uniform(low, high)

        # Evaluate offspring
        offspring_fitness = [func(individual) for individual in offspring]
        eval_count += len(offspring)

        # Combine and select the best individuals from the combined pool
        combined_population = population + offspring
        combined_fitness = fitness + offspring_fitness
        combined = sorted(zip(combined_population, combined_fitness), key=lambda x: x[1])
        population, fitness = zip(*combined[:population_size])
        population = list(map(list, population))
        fitness = list(fitness)

        # Update the best solution found
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]

    # Return fitness of the best found solution
    return best_fitness
#
#### Improvements Made:
#1. **Uniform Crossover**: Introduced a uniform crossover for potentially better exploration of the search space.
#   
#2. **Clear Handling**: Ensured conversions from tuples to lists after sorting.
#
#This enhances the algorithm's robustness and makes the evolutionary process potentially more effective in exploring the solution space.
