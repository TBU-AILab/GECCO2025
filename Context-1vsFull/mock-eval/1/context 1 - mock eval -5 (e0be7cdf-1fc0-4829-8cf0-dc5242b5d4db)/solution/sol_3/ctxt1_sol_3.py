import random

# Count the number of evaluations
eval_count = 0

def run(func, dim, bounds, max_evals):
    global eval_count
    eval_count = 0

    # Initialize variables
    best_solution = None
    best_fitness = float('inf')
    population_size = 30
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]

    # Define a helper function to evaluate the fitness
    def evaluate(individual):
        global eval_count
        if eval_count >= max_evals:
            return float('inf')
        fitness = func(individual)
        eval_count += 1
        return fitness

    # Evaluate the initial population
    fitnesses = [evaluate(individual) for individual in population]

    while eval_count < max_evals:
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
        
        new_fitnesses = [evaluate(individual) for individual in new_population]
        
        combined_population = population + new_population
        combined_fitnesses = fitnesses + new_fitnesses
        
        sorted_combined = sorted(zip(combined_population, combined_fitnesses), key=lambda x: x[1])
        
        population, fitnesses = zip(*sorted_combined[:population_size])
        population = [list(individual) for individual in population]
        fitnesses = list(fitnesses)

        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_solution = population[0]

        # Adding elitism by carrying best individual to next gen
        worst_index = fitnesses.index(max(fitnesses))
        population[worst_index] = best_solution
        fitnesses[worst_index] = best_fitness

    return best_solution, best_fitness, eval_count
