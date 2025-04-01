def run(func, dim, bounds, max_evals):
    # Initialize the population
    population_size = 20
    perturbation_strength = 0.1
    population = [([low + (high - low) * random() for low, high in bounds], None) for _ in range(population_size)]
    for i in range(population_size):
        population[i] = (population[i][0], func(population[i][0]))
    
    evals = population_size
    best = min(population, key=lambda x: x[1])[1]  # Get the best individual's fitness
    
    while evals < max_evals:
        offspring = []
        
        for i, (individual, fitness) in enumerate(population):
            # Generate a new solution by perturbing the current one
            new_solution = [max(bounds[d][0], min(bounds[d][1], individual[d] + perturbation_strength * (random() * 2 - 1))) for d in range(dim)]
            new_fitness = func(new_solution)
            evals += 1
            
            # Compare and select if the new solution is better
            if new_fitness < fitness:
                offspring.append((new_solution, new_fitness))
            else:
                offspring.append((individual, fitness))
        
        # Incorporate offspring into the next generation of population
        population = sorted(offspring, key=lambda x: x[1])[:population_size]
        best = min(population, key=lambda x: x[1])[1]
    
    return best

from random import random

# Example usage assuming an external func
def func_example(x):
    return sum(x_i**2 for x_i in x)

dimensionality = 5
bounds_example = [(-5, 5) for _ in range(dimensionality)]
maximum_evaluations = 100

best_fitness = run(func_example, dimensionality, bounds_example, maximum_evaluations)
print("Best Fitness Found:", best_fitness)
