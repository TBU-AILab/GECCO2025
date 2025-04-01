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

        # Elitism: carry the best individual to the next generation
        elite_count = 2
        elite_indices = sorted(range(population_size), key=lambda i: population_fitness[i])[:elite_count]
        new_population.extend([population[i] for i in elite_indices])

        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            selected = random.sample(range(population_size), tournament_size)
            parent1 = min((population[i] for i in selected), key=lambda ind: population_fitness[population.index(ind)])
            selected = random.sample(range(population_size), tournament_size)
            parent2 = min((population[i] for i in selected), key=lambda ind: population_fitness[population.index(ind)])
            
            # Arithmetic Crossover
            alpha = random.random()
            offspring = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]

            # Mutate the offspring
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

        # Update population and fitness list
        population = new_population[:]
        population_fitness = new_fitness[:]

    return best
