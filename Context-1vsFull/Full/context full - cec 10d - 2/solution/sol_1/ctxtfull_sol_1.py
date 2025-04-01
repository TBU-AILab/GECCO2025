import random

def run(func, dim, bounds, max_evals):
    best_position = None
    best_score = float('inf')
    
    # Initialize population
    population_size = 100  # Increased population size for diversity
    mutation_rate = 0.2  # Increased mutation rate for more exploration
    crossover_rate = 0.8  # Added crossover rate to control crossover operations
    population = [[random.uniform(low, high) for low, high in bounds] for _ in range(population_size)]
    
    # Helper function for evaluating the fitness
    def evaluate_population(pop):
        fitness_scores = []
        nonlocal evals
        for individual in pop:
            if evals >= max_evals:
                break
            fitness = func(individual)
            fitness_scores.append(fitness)
            evals += 1
        return fitness_scores
    
    # Main loop for evolutionary algorithm
    evals = 0
    while evals < max_evals:
        # Evaluate current population
        fitness_scores = evaluate_population(population)
        
        # Update best solution found
        for i, fitness in enumerate(fitness_scores):
            if fitness < best_score:
                best_score = fitness
                best_position = population[i]
        
        # Selection: Roulette Wheel Selection
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1/len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = [f / total_fitness for f in fitness_scores]
        
        # Create next generation with selection, crossover, and mutation
        next_generation = []
        for _ in range(population_size // 2):
            parents = random.choices(population, weights=probabilities, k=2)
            if random.random() < crossover_rate:
                # Simulated Binary Crossover (SBX)
                crossover_point = random.randint(1, dim-1)
                beta = random.random()
                child1 = [
                    beta * parents[0][i] + (1 - beta) * parents[1][i]
                    for i in range(dim)
                ]
                child2 = [
                    beta * parents[1][i] + (1 - beta) * parents[0][i]
                    for i in range(dim)
                ]
            else:
                child1, child2 = parents[0][:], parents[1][:]
            
            # Mutation
            for child in (child1, child2):
                if random.random() < mutation_rate:
                    mutation_idx = random.randint(0, dim-1)
                    child[mutation_idx] = random.uniform(bounds[mutation_idx][0], bounds[mutation_idx][1])
                    
            next_generation.extend([child1, child2])
        
        # Replace old population with the new one for the next iteration
        population = next_generation[:population_size]

    return best_score
