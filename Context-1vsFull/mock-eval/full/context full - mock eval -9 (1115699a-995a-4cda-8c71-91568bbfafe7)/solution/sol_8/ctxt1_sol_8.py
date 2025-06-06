import random

def run(func, dim, bounds, max_evals):
    # Initialize
    best = float('inf')
    best_solution = None
    eval_count = 0

    # Generate initial population
    population_size = 20
    population = [[random.uniform(low, high) for low, high in bounds] for _ in range(population_size)]
    
    # Evaluate initial population
    population_fitness = []
    for individual in population:
        fitness = func(individual)
        eval_count += 1
        population_fitness.append(fitness)
        if fitness < best:
            best = fitness
            best_solution = individual

    def adaptive_mutation(prob_base, eval_count, max_evals):
        return prob_base * (1 - eval_count / max_evals)

    # Algorithm main loop
    while eval_count < max_evals:
        new_population = []
        mutation_probability = adaptive_mutation(0.3, eval_count, max_evals)

        # Carry the best individuals to the next generation (Elitism)
        elite_count = max(1, population_size // 5)
        elite_indices = sorted(range(population_size), key=lambda i: population_fitness[i])[:elite_count]
        new_population.extend([population[i] for i in elite_indices])

        # Generate new solutions
        while len(new_population) < population_size:
            # Tournament selection for diversity
            tournament_size = 3
            indices1 = random.sample(range(population_size), tournament_size)
            indices2 = random.sample(range(population_size), tournament_size)
            parent1 = min(indices1, key=lambda i: population_fitness[i])
            parent2 = min(indices2, key=lambda i: population_fitness[i])
            
            # Blend Crossover: blend parent genes
            blend_alpha = 0.5
            offspring = [(blend_alpha * x + (1 - blend_alpha) * y) for x, y in zip(population[parent1], population[parent2])]

            # Mutate the offspring with adaptive probability
            if random.random() < mutation_probability:
                mutation_index = random.randint(0, dim - 1)
                offspring[mutation_index] = random.uniform(bounds[mutation_index][0], bounds[mutation_index][1])

            new_population.append(offspring)

        # Evaluate the new population
        new_fitness = []
        for individual in new_population:
            fitness = func(individual)
            eval_count += 1
            new_fitness.append(fitness)
            if fitness < best:
                best = fitness
                best_solution = individual

            if eval_count >= max_evals:
                break

        # Replace the old population with the new one
        population = new_population
        population_fitness = new_fitness

    return best
