import random

def run(func, dim, bounds, max_evals):
    best_position = None
    best_score = float('inf')
    
    # Initialize population with greater diversity
    population_size = 100
    mutation_rate = 0.3  # Higher mutation rate for better exploration
    crossover_rate = 0.9  # High crossover rate for diversity
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
        
        # Calculate probabilities with a focus on minimization
        adjusted_fitness = [1.0 / (f + 1e-9) for f in fitness_scores]  # Adding a small constant to prevent division by zero
        total_adjusted_fitness = sum(adjusted_fitness)
        probabilities = [f / total_adjusted_fitness for f in adjusted_fitness] if total_adjusted_fitness > 0 else [1.0 / population_size] * population_size
        
        # Create next generation with selection, crossover, and mutation
        next_generation = elites[:]
        while len(next_generation) < population_size:
            parents = random.choices(population, weights=probabilities, k=2)
            if random.random() < crossover_rate:
                # Blend crossover for more nuanced combinations
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
            
            # Apply mutation
            for child in (child1, child2):
                if random.random() < mutation_rate:
                    mutation_idx = random.randint(0, dim-1)
                    child[mutation_idx] = random.uniform(bounds[mutation_idx][0], bounds[mutation_idx][1])
                
                # Ensure only unique individuals to maintain diversity
                if child not in next_generation:
                    next_generation.append(child)
        
        # Update population
        population = next_generation[:population_size]

    return best_score
