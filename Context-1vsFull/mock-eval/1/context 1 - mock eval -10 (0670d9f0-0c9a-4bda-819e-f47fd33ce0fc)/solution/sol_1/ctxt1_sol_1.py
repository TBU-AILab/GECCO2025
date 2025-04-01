import random

def run(func, dim, bounds, max_evals):
    best_solution = None
    best_fitness = float('inf')
    evals = 0

    # Initialize population
    population_size = 20
    population = [
        [random.uniform(low, high) for low, high in bounds]
        for _ in range(population_size)
    ]

    def evaluate_solution(solution):
        nonlocal evals
        evals += 1
        return func(solution)

    # Evaluate initial population
    fitness_values = [
        evaluate_solution(individual) for individual in population
    ]

    # Update global best
    def update_global_best():
        nonlocal best_fitness, best_solution
        min_fitness_index = fitness_values.index(min(fitness_values))
        if fitness_values[min_fitness_index] < best_fitness:
            best_fitness = fitness_values[min_fitness_index]
            best_solution = population[min_fitness_index]
    
    update_global_best()

    # Main loop
    while evals < max_evals:
        # Parent selection using tournament selection for increased diversity
        def tournament_selection():
            tournament_size = 5
            selected = random.sample(population, tournament_size)
            selected_fitness = [evaluate_solution(indv) for indv in selected]
            return selected[selected_fitness.index(min(selected_fitness))]

        parents = [tournament_selection(), tournament_selection()]

        # Crossover
        crossover_point = random.randint(1, dim - 1)
        offspring1 = parents[0][:crossover_point] + parents[1][crossover_point:]
        offspring2 = parents[1][:crossover_point] + parents[0][crossover_point:]

        # Mutation with adaptive probability
        mutation_probability = 0.1 + 0.4 * (1 - evals / max_evals)
        def mutate(offspring):
            for i in range(dim):
                if random.random() < mutation_probability:
                    low, high = bounds[i]
                    offspring[i] = random.uniform(low, high)

        mutate(offspring1)
        mutate(offspring2)

        # Evaluate offsprings
        offspring_fitness1 = evaluate_solution(offspring1)
        offspring_fitness2 = evaluate_solution(offspring2)

        # Replace worst individuals in population
        def replace_worst(offspring, offspring_fitness):
            worst_idx = fitness_values.index(max(fitness_values))
            if offspring_fitness < fitness_values[worst_idx]:
                population[worst_idx] = offspring
                fitness_values[worst_idx] = offspring_fitness

        replace_worst(offspring1, offspring_fitness1)
        replace_worst(offspring2, offspring_fitness2)

        # Update global best
        update_global_best()

    return best_fitness
