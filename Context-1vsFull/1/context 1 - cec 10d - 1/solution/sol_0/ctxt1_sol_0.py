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
        # Tournament selection
        tournament_size = 3
        selected = []
        for _ in range(2):
            aspirants = random.sample(list(zip(population, fitness_values)), tournament_size)
            winner = min(aspirants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(parent1, parent2):
        # Single-point crossover
        point = random.randint(1, dim - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(individual, bounds):
        # Random reset mutation
        mutation_rate = 0.1
        new_individual = individual[:]
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                new_individual[i] = random.uniform(bounds[i][0], bounds[i][1])
        return new_individual

    import random

    # Parameters
    population_size = 20
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
