#Certainly! Let's enhance the algorithm by focusing on improving boundary handling, diversity, and efficiency while maintaining readability and a straightforward structure.
#
import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    best_params = None
    
    # Initialize a population of candidate solutions
    population_size = 50
    population = [create_random_solution(dim, bounds) for _ in range(population_size)]
    
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

        # Sort population by fitness (ascending order)
        fitnesses.sort(key=lambda x: x[1])
        # Select the best half of the population
        num_parents = population_size // 2
        parents = [f[0] for f in fitnesses[:num_parents]]

        # Generate new population through crossover and mutation
        new_population = crossover_and_mutate(parents, bounds, population_size)

        # Replace old population with the new generation
        population = new_population

        # Introduce further diversity by adding new random solutions periodically
        if evals % (max_evals // 10) == 0:  # every 10% of the evaluation limit
            diversity_injection(population, dim, bounds, int(population_size * 0.1))

    return best

def create_random_solution(dim, bounds):
    return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

def crossover_and_mutate(parents, bounds, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child1, child2 = crossover(parent1, parent2, bounds)
        new_population.append(mutate(child1, bounds))
        if len(new_population) < population_size:
            new_population.append(mutate(child2, bounds))
    return new_population

def crossover(parent1, parent2, bounds):
    # Adaptive Simulated Binary Crossover (SBX)
    child1, child2 = [], []
    eta = 2.0
    for i in range(len(parent1)):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
        
        c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
        c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        child1.append(min(max(c1, bounds[i][0]), bounds[i][1]))
        child2.append(min(max(c2, bounds[i][0]), bounds[i][1]))

    return child1, child2

def mutate(solution, bounds):
    mutation_rate = 0.1
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            perturbation = random.gauss(0, (bounds[i][1] - bounds[i][0]) * 0.05)
            solution[i] += perturbation
            solution[i] = min(max(solution[i], bounds[i][0]), bounds[i][1])
    return solution

def diversity_injection(population, dim, bounds, num_injections):
    for _ in range(num_injections):
        random_solution = create_random_solution(dim, bounds)
        index_to_replace = random.randint(0, len(population) - 1)
        population[index_to_replace] = random_solution
#
#### Key Enhancements:
#
#1. **Adaptive SBX Crossover**: Utilizes a distribution index to maintain variance, producing offspring that are suitably diverse.
#
#2. **Efficient Diversity Injection**: Conducts periodic diversity injections based on evaluations, ensuring that new genetic material is introduced without excessive disruption.
#
#3. **Boundary Management**: Consistently checks and adjusts solutions to remain within the defined bounds, preventing boundary violations.
#
#These improvements solidify the algorithm's structure and ensure ongoing balance between exploration and exploitation, improving resilience in diverse optimization landscapes.
