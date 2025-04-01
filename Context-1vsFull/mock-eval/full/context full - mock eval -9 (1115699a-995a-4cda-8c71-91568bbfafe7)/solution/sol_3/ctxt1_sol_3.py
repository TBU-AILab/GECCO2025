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
        mutation_probability = adaptive_mutation(0.3, eval_count, max_evals)  # Slightly higher base mutation rate for diversity

        # Tournament selection and crossover
        for _ in range(population_size):
            tournament_size = 3  # Small tournament size for faster convergence
            selected_indices = random.sample(range(population_size), tournament_size)
            parent1 = min((population[i] for i in selected_indices), key=lambda ind: population_fitness[population.index(ind)])
            selected_indices = random.sample(range(population_size), tournament_size)
            parent2 = min((population[i] for i in selected_indices), key=lambda ind: population_fitness[population.index(ind)])
            
            # Uniform crossover
            offspring = [(p1 if random.random() < 0.5 else p2) for p1, p2 in zip(parent1, parent2)]

            # Mutate the offspring
            if random.random() < mutation_probability:
                mutation_index = random.randint(0, dim - 1)
                offspring[mutation_index] = random.uniform(bounds[mutation_index][0], bounds[mutation_index][1])

            new_population.append(offspring)

        # Evaluate the new population and keep track of evaluations
        for individual in new_population:
            fitness = func(individual)
            eval_count += 1
            if fitness < best:
                best = fitness
                best_solution = individual

            # Greedy replacement: replace the worst in the current population with the new individual if it's better
            worst_fitness_index = population_fitness.index(max(population_fitness))
            if fitness < population_fitness[worst_fitness_index]:
                population[worst_fitness_index] = individual
                population_fitness[worst_fitness_index] = fitness

            if eval_count >= max_evals:
                break

    return best
