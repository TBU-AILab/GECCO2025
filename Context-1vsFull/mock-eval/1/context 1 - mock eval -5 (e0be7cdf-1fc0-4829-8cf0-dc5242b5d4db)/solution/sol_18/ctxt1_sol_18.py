import random

# Global evaluation counter
eval_count = 0

def run(func, dim, bounds, max_evals):
    global eval_count
    eval_count = 0

    # Initial parameters
    population_size = 30
    crossover_rate = 0.7
    mutation_rate = 0.1

    # Initialize the population
    population = [
        [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] 
        for _ in range(population_size)
    ]

    # Evaluate the initial population
    fitnesses = [evaluate_func(func, individual) for individual in population]

    # Main optimization loop
    while eval_count < max_evals:
        new_population = []

        for _ in range(population_size // 2):
            # Select parents
            parents = select_parents(population, fitnesses)
            # Crossover
            offspring = crossover(parents, crossover_rate, dim)
            # Mutation
            offspring = [mutate(child, mutation_rate, dim, bounds) for child in offspring]
            new_population.extend(offspring)

        # Evaluate new population
        new_fitnesses = [evaluate_func(func, individual) for individual in new_population]

        # Select the next generation
        population, fitnesses = selection(population, fitnesses, new_population, new_fitnesses)

    # Best solution
    best_index = fitnesses.index(min(fitnesses))
    best_solution = population[best_index]
    best_fitness = fitnesses[best_index]

    return best_solution, best_fitness, eval_count

def evaluate_func(func, individual):
    global eval_count
    eval_count += 1
    return func(individual)

def select_parents(population, fitnesses):
    tournament_size = 3
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1])
    return selected[0][0], selected[1][0]

def crossover(parents, crossover_rate, dim):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, dim - 1)
        child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
        child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
        return [child1, child2]
    else:
        return [list(parents[0]), list(parents[1])]

def mutate(individual, mutation_rate, dim, bounds):
    if random.random() < mutation_rate:
        mutate_index = random.randrange(dim)
        individual[mutate_index] = random.uniform(bounds[mutate_index][0], bounds[mutate_index][1])
    return individual

def selection(old_population, old_fitnesses, new_population, new_fitnesses):
    combined_population = old_population + new_population
    combined_fitnesses = old_fitnesses + new_fitnesses

    sorted_combined = sorted(zip(combined_population, combined_fitnesses), key=lambda x: x[1])

    selected_population, selected_fitnesses = zip(*sorted_combined[:len(old_population)])
    selected_population = [list(individual) for individual in selected_population]
    selected_fitnesses = list(selected_fitnesses)

    return selected_population, selected_fitnesses

# Test function: Inverted Sphere
def inverted_sphere(individual):
    return -sum(x**2 for x in individual)

# Example usage
dim = 5
bounds = [(-5.0, 5.0) for _ in range(dim)]
max_evals = 1000

best_solution, best_fitness, evals_used = run(inverted_sphere, dim, bounds, max_evals)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
print("Evaluations Used:", evals_used)
