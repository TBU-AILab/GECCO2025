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
        elite_count = max(1, population_size // 10)
        elite_indices = sorted(range(population_size), key=lambda i: population_fitness[i])[:elite_count]
        new_population.extend([population[i] for i in elite_indices])

        while len(new_population) < population_size:
            # Tournament selection with higher diversity
            tournament_size = 5
            selected_indices = random.sample(range(population_size), tournament_size)
            parent1 = min(selected_indices, key=lambda i: population_fitness[i])
            selected_indices = random.sample(range(population_size), tournament_size)
            parent2 = min(selected_indices, key=lambda i: population_fitness[i])
            
            # Simulated Binary Crossover (SBX)
            eta_c = 2.0
            offspring = []
            for x1, x2 in zip(population[parent1], population[parent2]):
                if random.random() <= 0.5:
                    if abs(x1 - x2) > 1e-14:
                        beta = 1.0 + (2.0 * min(x1 - bounds[0][0], x2 - bounds[0][0]) / abs(x2 - x1))
                        alpha = 2.0 - beta**-(eta_c + 1.0)
                        if random.random() <= 1.0 / alpha:
                            beta_q = (random.random() * alpha)**(1.0 / (eta_c + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - alpha * random.random()))**(1.0 / (eta_c + 1.0))
                        offspring_val = 0.5 * ((x1 + x2) - beta_q * abs(x1 - x2))
                        offspring.append(min(max(offspring_val, bounds[0][0]), bounds[0][1]))
                    else:
                        offspring.append(x1)
                else:
                    offspring.append(x1)

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
