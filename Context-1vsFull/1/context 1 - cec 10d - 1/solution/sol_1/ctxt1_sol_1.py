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
        # Tournament selection with dynamic tournament size based on population diversity
        diversity = len(set(map(tuple, population)))
        tournament_size = max(2, diversity // 10)  # Dynamic selection pressure
        selected = []
        for _ in range(2):
            aspirants = random.sample(list(zip(population, fitness_values)), min(tournament_size, len(population)))
            winner = min(aspirants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(parent1, parent2):
        # Blend crossover (BLX-0.5) for better exploration
        alpha = 0.5
        child1 = []
        child2 = []
        for p1, p2 in zip(parent1, parent2):
            d = abs(p1 - p2)
            lower = min(p1, p2) - alpha * d
            upper = max(p1, p2) + alpha * d
            child1.append(random.uniform(lower, upper))
            child2.append(random.uniform(lower, upper))
        return child1, child2

    def mutate(individual, bounds):
        # Gaussian mutation for more controlled variation
        mutation_rate = 0.1
        mutation_sd = 0.1  # Standard deviation for mutation
        new_individual = individual[:]
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                new_individual[i] += random.gauss(0, mutation_sd)
                new_individual[i] = max(bounds[i][0], min(new_individual[i], bounds[i][1]))
        return new_individual

    import random

    # Parameters
    population_size = 30
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
            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1, bounds))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, bounds))

        population = new_population

    return best
