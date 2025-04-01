def run(func, dim, bounds, max_evals):
    def initialize_population(size, dim, bounds):
        return [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(size)]

    def evaluate_population(population):
        evals = 0
        fitness_values = []
        for individual in population:
            fitness = func(individual)
            evals += 1
            fitness_values.append(fitness)
        return fitness_values, evals

    def select_parents(population, fitness_values):
        # Tournament selection with dynamic tournament size based on diversity
        diversity = len(set(map(tuple, population)))
        tournament_size = max(2, diversity // 10)  # Dynamic selection pressure
        selected = []
        for _ in range(2):
            aspirants = random.sample(list(zip(population, fitness_values)), min(tournament_size, len(population)))
            winner = min(aspirants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(parent1, parent2):
        # Arithmetic crossover with potential adaptive blending factor
        alpha = 0.5
        child = [(alpha * p1 + (1 - alpha) * p2) for p1, p2 in zip(parent1, parent2)]
        return child

    def mutate(individual, bounds):
        # Adaptive Gaussian mutation with annealing mutation strength
        initial_mutation_rate = 0.2
        mutation_decay = 0.95
        current_rate = max(0.01, initial_mutation_rate * mutation_decay)
        new_individual = individual[:]
        for i in range(len(individual)):
            if random.random() < current_rate:
                sd = (bounds[i][1] - bounds[i][0]) * 0.1
                new_individual[i] += random.gauss(0, sd)
                new_individual[i] = max(bounds[i][0], min(new_individual[i], bounds[i][1]))
        return new_individual

    import random

    # Parameters
    population_size = 50
    population = initialize_population(population_size, dim, bounds)
    best = float('inf')
    best_individual = None
    total_evals = 0

    # Algorithm body
    while total_evals < max_evals:
        fitness_values, evals = evaluate_population(population)
        total_evals += evals
        
        # Update best solution
        for i, fitness in enumerate(fitness_values):
            if fitness < best:
                best = fitness
                best_individual = population[i]

        new_population = []
        
        while len(new_population) < population_size:
            parents = select_parents(population, fitness_values)
            child = mutate(crossover(parents[0], parents[1]), bounds)
            new_population.append(child)

        population = new_population

    return best
