def run(func, dim, bounds, max_evals):
    import random

    # Define helper function for boundary checking
    def clip(params, bounds):
        return [max(min(params[i], bounds[i][1]), bounds[i][0]) for i in range(dim)]

    # Define the cost function counting mechanism
    class CostFunctionEvaluator:
        def __init__(self, func):
            self.func = func
            self.evals = 0

        def evaluate(self, solution):
            self.evals += 1
            return self.func(solution)

    # Wrapper for counting evaluations
    cost_evaluator = CostFunctionEvaluator(func)

    # Initialization of parameters
    population_size = 100
    mutation_strength = 0.1
    crossover_rate = 0.7
    elite_ratio = 0.1
    num_elites = max(1, int(population_size * elite_ratio))

    # Initialize population
    population = [
        clip([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)], bounds)
        for _ in range(population_size)
    ]

    # Evaluate initial population
    population_fitness = [cost_evaluator.evaluate(ind) for ind in population]

    # Main loop
    while cost_evaluator.evals < max_evals:
        # Sort population based on fitness
        sorted_indices = sorted(range(population_size), key=lambda i: population_fitness[i])

        # Select elites
        elites = [population[i] for i in sorted_indices[:num_elites]]

        # Generate new population using crossover and mutation
        new_population = elites.copy()
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                # Perform crossover
                parent1, parent2 = random.sample(elites, 2)
                crossover_point = random.randint(1, dim - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                # Mutation
                parent = random.choice(elites)
                child = [
                    parent[i] + random.gauss(0, mutation_strength) * (bounds[i][1] - bounds[i][0]) 
                    for i in range(dim)
                ]
            
            child = clip(child, bounds)
            new_population.append(child)

        # Evaluate new population
        population = new_population[:population_size]
        population_fitness = [cost_evaluator.evaluate(ind) for ind in population]

        # Early stopping if evaluations reach max_evals
        if cost_evaluator.evals >= max_evals:
            break

    # Extract best solution found
    best_idx = min(range(population_size), key=lambda i: population_fitness[i])
    best_fitness = population_fitness[best_idx]
    best_solution = population[best_idx]

    return best_fitness, best_solution
