def run(func, dim, bounds, max_evals):
    import random
    
    # Define helper function to ensure parameters are within bounds
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
    
    # Initialization of optimization parameters
    population_size = 50
    mutation_strength = 0.1
    crossover_rate = 0.8
    elite_ratio = 0.2
    num_elites = max(1, int(population_size * elite_ratio))
    
    # Initialize population
    population = [
        clip([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)], bounds)
        for _ in range(population_size)
    ]
    
    # Evaluate initial population
    population_fitness = [cost_evaluator.evaluate(ind) for ind in population]
    
    # Main optimization loop
    while cost_evaluator.evals < max_evals:
        # Sort population by fitness (lower is better)
        sorted_indices = sorted(range(population_size), key=lambda i: population_fitness[i])
        
        # Extract elites
        elites = [population[i] for i in sorted_indices[:num_elites]]
        
        # Generate new population using crossover and mutation
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            # Crossover operation
            if random.random() < crossover_rate:
                alpha = random.random()
                child = [
                    alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(dim)
                ]
            else:
                child = parent1.copy()
            
            # Mutation operation
            for i in range(dim):
                if random.random() < mutation_strength:
                    scale = (bounds[i][1] - bounds[i][0]) * 0.1
                    child[i] += random.gauss(0, scale)
            
            child = clip(child, bounds)
            new_population.append(child)
        
        # Replace old population with the new generation
        population = new_population[:population_size]
        
        # Evaluate new population
        population_fitness = [cost_evaluator.evaluate(ind) for ind in population]
    
    # Extract best solution found
    best_idx = min(range(population_size), key=lambda i: population_fitness[i])
    best_fitness = population_fitness[best_idx]
    best_solution = population[best_idx]
    
    return best_fitness, best_solution

# Example usage with a simple quadratic function
def example_func(solution):
    """Example cost function."""
    return sum(x ** 2 for x in solution)

bounds = [(-5, 5)] * 5
best_value, best_vector = run(example_func, 5, bounds, 1000)
print("Best value:", best_value)
print("Best solution:", best_vector)
#
#In this version:
#- Maintained population diversity with increased crossover and acceptable mutation rate.
#- Improved consistency in selecting parents for crossover from elite members.
#- Ensured the parameter clipping and fitness evaluation mechanisms remain robust.
