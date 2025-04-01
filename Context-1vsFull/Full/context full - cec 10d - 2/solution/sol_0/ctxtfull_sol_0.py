import random

def run(func, dim, bounds, max_evals):
    best_position = None
    best_score = float('inf')
    
    # Initialize population
    population_size = 50  # You can adjust this
    mutation_rate = 0.1  # Rate of mutation
    population = [[random.uniform(low, high) for low, high in bounds] for _ in range(population_size)]
    
    # Main loop for evolutionary algorithm
    evals = 0
    while evals < max_evals:
        # Evaluate current population
        fitness_scores = []
        for individual in population:
            fitness = func(individual)
            fitness_scores.append(fitness)
            evals += 1
            
            if fitness < best_score:
                best_score = fitness
                best_position = individual
        
        # Selection: Tournament Selection
        next_generation = []
        for _ in range(population_size):
            contestants = random.sample(population, 3)
            contestant_fitness = [fitness_scores[population.index(c)] for c in contestants]
            winner = contestants[contestant_fitness.index(min(contestant_fitness))]
            next_generation.append(winner[:])  # Add to next generation

        # Crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = next_generation[i]
            parent2 = next_generation[i+1 if i+1 < population_size else 0]
            
            # Single-point crossover
            crossover_point = random.randint(1, dim-1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            # Mutation
            for child in (child1, child2):
                if random.random() < mutation_rate:
                    mutation_idx = random.randint(0, dim-1)
                    child[mutation_idx] = random.uniform(bounds[mutation_idx][0], bounds[mutation_idx][1])
                
                new_population.append(child)
        
        # Replace the old population with the new one
        population = new_population
    
    return best_score
