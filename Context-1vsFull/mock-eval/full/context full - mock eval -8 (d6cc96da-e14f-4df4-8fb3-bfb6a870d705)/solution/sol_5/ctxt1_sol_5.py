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
        top_fraction = 0.5
        num_parents = int(population_size * top_fraction)
        parents = [f[0] for f in fitnesses[:num_parents]]

        # Generate new population through crossover and mutation
        new_population = crossover_and_mutate(parents, bounds, population_size)

        # Replace the worst candidates
        population[num_parents:] = new_population[num_parents:]

        # Introduce diversity by adding random solutions
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
    # Simulated Binary Crossover (SBX)
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            beta = 2.0 * random.random()
        else:
            beta = 1.0 / (2.0 * random.random())
        
        beta = beta ** (1.0 / 3.0)
        c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
        c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        c1 = min(max(c1, bounds[i][0]), bounds[i][1])
        c2 = min(max(c2, bounds[i][0]), bounds[i][1])
        child.append(c1 if random.random() < 0.5 else c2)
    return child

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
#1. **Simulated Binary Crossover (SBX)**: This crossover technique mimics natural genetics and creates diversity in offspring by a controlled variation called the distribution index. It is a popular choice in evolutionary algorithms for its effectiveness in real-parameter optimizations.
#
#2. **Balanced Parent Selection**: The algorithm now selects the best half of the current population efficiently, ensuring that only the fittest parents contribute to the next generation.
#
#3. **Effective Replacement and Diversity**: By replacing only the worst half with the newly generated population and introducing random solutions strategically, enhancement in search capability and avoidance of premature convergence are achieved.
#
#These changes are focused on enhancing the exploration and exploitation balance by maintaining genetic diversity and leveraging effective recombination strategies.
