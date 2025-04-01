import random

def run(func, dim, bounds, max_evals):
    # Population parameters
    population_size = 20
    mutation_rate = 0.1
    tournament_size = 3

    # Initialize the best solution to infinity
    best = float('inf')
    eval_count = 0

    # Function to initialize the population
    def initialize_population(size, dim, bounds):
        return [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(size)]

    # Function to evaluate a population
    def evaluate_population(population):
        nonlocal eval_count
        fitness = []
        for individual in population:
            if eval_count < max_evals:
                fitness_value = func(individual)
                eval_count += 1
                fitness.append((fitness_value, individual))
            else:
                break
        return sorted(fitness, key=lambda x: x[0])

    # Function to perform mutation
    def mutate(individual, bounds, rate):
        return [random.uniform(bounds[d][0], bounds[d][1]) if random.random() < rate else individual[d]
                for d in range(len(individual))]

    # Function for recombination (crossover)
    def recombine(parent1, parent2):
        return [(p1 + p2) / 2 if random.random() > 0.5 else p1 for p1, p2 in zip(parent1, parent2)]

    # Function for tournament selection
    def tournament_selection(evaluated_population, size):
        selected = random.sample(evaluated_population, min(size, len(evaluated_population)))
        return min(selected, key=lambda x: x[0])[1]

    # Initialize population
    population = initialize_population(population_size, dim, bounds)

    # Main optimization loop
    while eval_count < max_evals:
        # Evaluate current population
        evaluated_population = evaluate_population(population)
        
        # Check if the evaluations were exhausted
        if not evaluated_population:
            break

        # Update the best solution found
        current_best = evaluated_population[0][0]
        if current_best < best:
            best = current_best

        # Generate the next generation
        next_generation = []
        while len(next_generation) < population_size:
            # Select parents using tournament selection
            parent1 = tournament_selection(evaluated_population, tournament_size)
            parent2 = tournament_selection(evaluated_population, tournament_size)

            # Recombine and mutate to create offspring
            offspring1 = mutate(recombine(parent1, parent2), bounds, mutation_rate)
            offspring2 = mutate(recombine(parent2, parent1), bounds, mutation_rate)

            # Add offspring to the next generation
            next_generation.extend([offspring1, offspring2])

        # Trim the generation if it exceeds the population size
        population = next_generation[:population_size]

    # Return the fitness of the best found solution
    return best
