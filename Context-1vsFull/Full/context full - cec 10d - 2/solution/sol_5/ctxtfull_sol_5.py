import random

def run(func, dim, bounds, max_evals):
    best_position = None
    best_score = float('inf')
    
    # Initialize population with diversity consideration
    population_size = 100
    mutation_rate = 0.3  # Increase mutation rate for broader exploration
    crossover_rate = 0.9  # Ensure most crossover to increase exploration
    elite_rate = 0.1  # Introduce elitism
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
        
        # Handle situation when population size does not match fitness_scores length
        if len(fitness_scores) < population_size:
            population = population[:len(fitness_scores)]

        # Select elites
        elites = [x for _, x in sorted(zip(fitness_scores, population), key=lambda x: x[0])[:elite_count]]
        
        # Roulette wheel selection adjusted for minimization
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1/len(fitness_scores)] * len(fitness_scores)
        else:
            min_fitness = min(fitness_scores)
            adjusted_fitness = [1 / (f - min_fitness + 1) for f in fitness_scores]
            total_adjusted_fitness = sum(adjusted_fitness)
            probabilities = [f / total_adjusted_fitness for f in adjusted_fitness]
        
        # Create next generation with selected parents
        next_generation = elites[:]
        while len(next_generation) < population_size:
            parents = random.choices(population, weights=probabilities, k=2)
            if random.random() < crossover_rate:
                # Simulated Binary Crossover (SBX)
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
            
            # Apply mutation
            for child in (child1, child2):
                if random.random() < mutation_rate:
                    mutation_idx = random.randint(0, dim-1)
                    child[mutation_idx] = random.uniform(bounds[mutation_idx][0], bounds[mutation_idx][1])
                
                next_generation.append(child)
        
        # Update population
        population = next_generation[:population_size]

    return best_score
