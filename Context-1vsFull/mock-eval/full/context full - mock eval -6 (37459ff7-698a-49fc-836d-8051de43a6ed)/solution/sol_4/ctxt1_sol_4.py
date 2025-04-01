import random

def run(func, dim, bounds, max_evals):
    # Initialize the best solution to infinity
    best = float('inf')
    best_params = None

    # Initialize a counter for cost function evaluations
    eval_count = 0

    # Define population and mutation settings
    population_size = 20
    mutation_rate = 0.1
    tournament_size = 3

    # Function to initialize the population
    def initialize_population(size, dim, bounds):
        return [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(size)]

    # Function to evaluate a population
    def evaluate_population(population):
        nonlocal eval_count
        fitness = []
        for individual in population:
            if eval_count >= max_evals:
                return fitness
            fitness_value = func(individual)
            eval_count += 1
            fitness.append((fitness_value, individual))
        return sorted(fitness, key=lambda x: x[0])

    # Function to perform mutation
    def mutate(individual, bounds, rate):
        return [random.uniform(bounds[d][0], bounds[d][1]) if random.random() < rate else individual[d]
                for d in range(len(individual))]

    # Function for recombination (crossover)
    def recombine(parent1, parent2):
        return [parent1[d] if random.random() > 0.5 else parent2[d] for d in range(len(parent1))]

    # Function for tournament selection
    def tournament_selection(evaluated_population, size):
        selected = random.sample(evaluated_population, min(size, len(evaluated_population)))
        return min(selected, key=lambda x: x[0])[1]

    # Initialize population
    population = initialize_population(population_size, dim, bounds)

    # Main optimization loop
    while eval_count < max_evals:
        # Evaluate current population, ensuring evaluations have not been exceeded
        evaluated_population = evaluate_population(population)
        if not evaluated_population:
            break

        # Update the best solution
        if evaluated_population[0][0] < best:
            best, best_params = evaluated_population[0]

        # Generate the next generation
        next_generation = []
        while len(next_generation) < population_size and eval_count < max_evals:
            # Select parents
            parent1 = tournament_selection(evaluated_population, tournament_size)
            parent2 = tournament_selection(evaluated_population, tournament_size)

            # Recombine parents to create offspring
            child1 = recombine(parent1, parent2)
            child2 = recombine(parent2, parent1)

            # Apply mutation to offspring
            child1 = mutate(child1, bounds, mutation_rate)
            child2 = mutate(child2, bounds, mutation_rate)

            # Add to the next generation
            next_generation.extend([child1, child2])

        # Update population
        population = next_generation[:population_size]

    # Return the fitness of the best found solution
    return best
