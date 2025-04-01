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
        total_fitness = sum(fitness_values)
        selection_probs = [(total_fitness - f) / total_fitness for f in fitness_values]
        parents = random.choices(population, weights=selection_probs, k=2)
        return parents

    def crossover(parent1, parent2):
        # Simulated binary crossover (SBX)
        eta_c = 2.0
        child = []
        for p1, p2 in zip(parent1, parent2):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            child.append(c1 if random.random() < 0.5 else c2)
        return child

    def mutate(individual, bounds):
        mutation_prob = 0.1
        eta_m = 20.0
        child = []
        for i in range(len(individual)):
            if random.random() < mutation_prob:
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                new_val = individual[i] + delta * (bounds[i][1] - bounds[i][0])
                new_val = max(bounds[i][0], min(new_val, bounds[i][1]))
            else:
                new_val = individual[i]
            child.append(new_val)
        return child

    import random

    # Enhanced Parameters
    population_size = 100  # Increase population to explore more solutions
    elitism_rate = 0.1
    elite_size = int(elitism_rate * population_size)
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

        sorted_population = [x for _, x in sorted(zip(fitness_values, population))]
        new_population = sorted_population[:elite_size]  # Preserve elite solutions
        
        while len(new_population) < population_size:
            parents = select_parents(population, fitness_values)
            child = mutate(crossover(parents[0], parents[1]), bounds)
            new_population.append(child)

        population = new_population

    return best
