import random

def run(func, dim, bounds, max_evals):
    # Initialize best solution
    best = float('inf')
    best_params = None

    # Initialize a population of candidate solutions
    population_size = 50
    population = [create_random_solution(bounds) for _ in range(population_size)]
    
    evals = 0

    # Main loop of the algorithm
    while evals < max_evals:
        # Evaluate the population
        for params in population:
            if evals >= max_evals:
                break
            fitness = func(params)
            evals += 1

            # Update the best solution found
            if fitness < best:
                best = fitness
                best_params = params
        
        # Select parents for crossover
        parents = select_parents(population, func)

        # Generate new population through crossover and mutation
        new_population = crossover_and_mutate(parents, bounds, population_size)
        population = new_population

    return best

def create_random_solution(bounds):
    return [random.uniform(low, high) for low, high in bounds]

def select_parents(population, func):
    # Tournament selection
    tournament_size = 3
    selected_parents = []
    for _ in range(len(population)):
        candidates = random.sample(population, tournament_size)
        candidates.sort(key=func)
        selected_parents.append(candidates[0])
    return selected_parents

def crossover_and_mutate(parents, bounds, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child, bounds)
        new_population.append(child)
    return new_population

def crossover(parent1, parent2):
    # Single-point crossover
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(solution, bounds):
    # Gaussian mutation
    mutation_rate = 0.1 
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] += random.gauss(0, 1)
            # Ensure the mutation is within bounds
            solution[i] = min(max(solution[i], bounds[i][0]), bounds[i][1])
    return solution
