import random

def example_cost_function(x):
    """Example objective function to be minimized."""
    return sum(x_i**2 for x_i in x)

def constrained_optimizer(func, dimensionality, bounds, max_evals):
    population_size = 20
    perturbation_strength = 0.1
    
    # Initialize population with random solutions
    population = [
        ([random.uniform(low, high) for low, high in bounds], None)
        for _ in range(population_size)
    ]
    
    evals = 0

    # Evaluate initial population
    for i in range(population_size):
        individual = population[i][0]
        population[i] = (individual, func(individual))
        evals += 1
    
    best_fitness = min(population, key=lambda x: x[1])[1]
    
    while evals < max_evals:
        offspring = []
        
        for individual, fitness in population:
            # Perturb current individual to generate new solution
            new_solution = [
                max(bounds[d][0], min(bounds[d][1], individual[d] + perturbation_strength * (random.random() * 2 - 1)))
                for d in range(dimensionality)
            ]
            new_fitness = func(new_solution)
            evals += 1
            
            # Select better solution
            if new_fitness < fitness:
                offspring.append((new_solution, new_fitness))
            else:
                offspring.append((individual, fitness))
        
        # Form new population with the best solutions
        population = sorted(offspring, key=lambda x: x[1])[:population_size]
        best_fitness = min(population, key=lambda x: x[1])[1]
    
    return best_fitness

def run():
    dimensionality = 5
    bounds = [(-5, 5) for _ in range(dimensionality)]
    max_evaluations = 100
    
    best_fitness = constrained_optimizer(example_cost_function, dimensionality, bounds, max_evaluations)
    print("Best Fitness Found:", best_fitness)

if __name__ == "__main__":
    run()
