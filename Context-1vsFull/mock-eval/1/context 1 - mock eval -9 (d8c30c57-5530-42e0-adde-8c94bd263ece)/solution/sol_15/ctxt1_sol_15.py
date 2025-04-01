import random

class MetaheuristicOptimizer:
    def __init__(self, func, dimensionality, bounds, max_evals):
        self.func = func
        self.dimensionality = dimensionality
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0

    def evaluate(self, solution):
        self.evals += 1
        return self.func(solution)

    def optimize(self):
        population_size = 20
        perturbation_strength = 0.1
        
        # Initialize population with random solutions
        population = [
            ([random.uniform(low, high) for low, high in self.bounds], None)
            for _ in range(population_size)
        ]
        
        # Evaluate initial population
        for i in range(population_size):
            individual = population[i][0]
            population[i] = (individual, self.evaluate(individual))
        
        best_fitness = min(population, key=lambda x: x[1])[1]
        
        while self.evals < self.max_evals:
            offspring = []
            
            for individual, fitness in population:
                # Perturb current individual to generate new solution
                new_solution = [
                    max(self.bounds[d][0], min(self.bounds[d][1], individual[d] + perturbation_strength * (random.random() * 2 - 1)))
                    for d in range(self.dimensionality)
                ]
                new_fitness = self.evaluate(new_solution)
                
                # Select better solution
                if new_fitness < fitness:
                    offspring.append((new_solution, new_fitness))
                else:
                    offspring.append((individual, fitness))
            
            # Form new population with the best solutions
            population = sorted(offspring, key=lambda x: x[1])[:population_size]
            best_fitness = min(population, key=lambda x: x[1])[1]
        
        return best_fitness

def example_cost_function(x):
    """Example objective function to be minimized."""
    return sum(x_i**2 for x_i in x)

def run(func, dimensionality, bounds, max_evaluations):
    optimizer = MetaheuristicOptimizer(func, dimensionality, bounds, max_evaluations)
    best_fitness = optimizer.optimize()
    print("Best Fitness Found:", best_fitness)

if __name__ == "__main__":
    dimensionality = 5
    bounds = [(-5, 5) for _ in range(dimensionality)]
    max_evaluations = 100
    
    run(example_cost_function, dimensionality, bounds, max_evaluations)
