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
        parents = [f[0] for f in fitnesses[:population_size // 2]]

        # Generate new population through crossover and mutation
        new_population = crossover_and_mutate(parents, bounds, population_size)

        # Replace the worst candidates with the newly injected random solutions
        population[:population_size // 2] = new_population
        diversity_injection(population, bounds, int(population_size * 0.1))

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
    # Arithmetic crossover with adaptive alpha
    alpha = random.uniform(0.4, 0.6)
    return [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]

def mutate(solution, bounds):
    mutation_rate = 0.2
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            perturbation = random.gauss(0, (bounds[i][1] - bounds[i][0]) * 0.05)
            solution[i] += perturbation
            # Ensure the mutation is within bounds
            solution[i] = min(max(solution[i], bounds[i][0]), bounds[i][1])
    return solution

def diversity_injection(population, bounds, num_injections):
    for _ in range(num_injections):
        random_solution = create_random_solution(bounds)
        index_to_replace = random.randint(len(population) // 2, len(population) - 1)
        population[index_to_replace] = random_solution
#
#### Key Improvements:
#1. **Adaptive Arithmetic Crossover**: The alpha parameter for crossover now ranges from 0.4 to 0.6 to maintain a balanced combination of parent traits dynamically.
#
#2. **Enhanced Diversity Injection**: Random solutions are injected into the worst half of the population, preventing stagnation by ensuring new genetic material is introduced consistently.
#
#3. **Balanced Replacement Strategy**: The solution now more evenly balances exploitation and exploration by combining the best solutions with new mutations and random injections.
#
#These improvements are aimed at fostering a robust search capability while maintaining diversity, allowing the algorithm to potentially escape local minima and better approximate the global optimum.
