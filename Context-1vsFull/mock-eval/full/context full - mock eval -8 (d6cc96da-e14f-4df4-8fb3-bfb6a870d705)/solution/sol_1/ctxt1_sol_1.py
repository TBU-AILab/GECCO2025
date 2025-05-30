import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    best_params = None
    
    # Initialize a population of candidate solutions
    population_size = 50
    population = [create_random_solution(bounds) for _ in range(population_size)]
    
    evals = 0

    # Main loop of the algorithm
    while evals < max_evals:
        # Evaluate the population
        fitnesses = []
        for params in population:
            if evals >= max_evals:
                break
            fitness = func(params)
            fitnesses.append((params, fitness))
            evals += 1

            # Update the best solution found
            if fitness < best:
                best = fitness
                best_params = params
        
        # Sort population by fitness
        fitnesses.sort(key=lambda x: x[1])
        # Select the best half of the population
        parents = [f[0] for f in fitnesses[:population_size//2]]

        # Generate new population through crossover and mutation
        new_population = crossover_and_mutate(parents, bounds, population_size)
        population = new_population

    return best

def create_random_solution(bounds):
    return [random.uniform(low, high) for low, high in bounds]

def crossover_and_mutate(parents, bounds, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child, bounds)
        new_population.append(child)
    return new_population

def crossover(parent1, parent2):
    # Uniform crossover
    return [parent1[i] if random.random() > 0.5 else parent2[i] for i in range(len(parent1))]

def mutate(solution, bounds):
    mutation_rate = 0.1
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] += random.gauss(0, (bounds[i][1] - bounds[i][0]) * 0.1)
            # Ensure the mutation is within bounds
            solution[i] = min(max(solution[i], bounds[i][0]), bounds[i][1])
    return solution
