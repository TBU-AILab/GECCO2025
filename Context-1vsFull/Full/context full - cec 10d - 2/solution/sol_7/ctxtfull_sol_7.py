import random

def run(func, dim, bounds, max_evals):
    best_position = None
    best_score = float('inf')
    
    # Initialize various parameters
    population_size = 150  # Larger population for better diversity
    mutation_rate = 0.2  # Balanced mutation rate
    crossover_rate = 0.9  # High crossover rate for broad exploration
    elite_rate = 0.1  # Preserve top solutions
    elite_count = int(population_size * elite_rate)
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
        
        # Select elites based on fitness scores
        elites = [x for _, x in sorted(zip(fitness_scores, population), key=lambda x: x[0])[:elite_count]]
        
        # Calculate probabilities with a sigmoid-based scaling for better selection pressure
        max_fitness = max(fitness_scores, default=1)
        probabilities = [1 / (1 + ((f - best_score) / max_fitness)) for f in fitness_scores]
        total_probabilities = sum(probabilities)
        probabilities = [p / total_probabilities for p in probabilities] if total_probabilities > 0 else [1.0 / population_size] * population_size
        
        # Create next generation with selection, crossover, and mutation
        next_generation = elites[:]
        while len(next_generation) < population_size:
            parents = random.choices(population, weights=probabilities, k=2)
            if random.random() < crossover_rate:
                # Arithmetic crossover for improved diversity
                alpha = random.random()
                child1 = [
                    alpha * parents[0][i] + (1 - alpha) * parents[1][i]
                    for i in range(dim)
                ]
                child2 = [
                    alpha * parents[1][i] + (1 - alpha) * parents[0][i]
                    for i in range(dim)
                ]
            else:
                child1, child2 = parents[0][:], parents[1][:]
            
            # Apply mutation step
            for child in (child1, child2):
                if random.random() < mutation_rate:
                    mutation_idx = random.randint(0, dim-1)
                    child[mutation_idx] = random.uniform(bounds[mutation_idx][0], bounds[mutation_idx][1])
                
                # Ensure diversity by allowing only unique individuals
                if len(next_generation) < population_size and child not in next_generation:
                    next_generation.append(child)
        
        # Update the population
        population = next_generation

    return best_score
