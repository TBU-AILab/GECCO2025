import random

def run(func, dim, bounds, max_evals):
    population_size = 20
    perturbation_strength = 0.1

    population = [
        ([low + (high - low) * random.random() for low, high in bounds], None)
        for _ in range(population_size)
    ]
    
    evals = 0

    for i in range(population_size):
        individual = population[i][0]
        population[i] = (individual, func(individual))
        evals += 1
    
    best_fitness = min(population, key=lambda x: x[1])[1]
    
    while evals < max_evals:
        offspring = []
        
        for individual, fitness in population:
            new_solution = [
                max(bounds[d][0], min(bounds[d][1], individual[d] + perturbation_strength * (random.random() * 2 - 1)))
                for d in range(dim)
            ]
            new_fitness = func(new_solution)
            evals += 1
            
            if new_fitness < fitness:
                offspring.append((new_solution, new_fitness))
            else:
                offspring.append((individual, fitness))
        
        population = sorted(offspring, key=lambda x: x[1])[:population_size]
        best_fitness = min(population, key=lambda x: x[1])[1]
    
    return best_fitness

def func_example(x):
    return sum(x_i**2 for x_i in x)

dimensionality = 5
bounds_example = [(-5, 5) for _ in range(dimensionality)]
maximum_evaluations = 100

best_fitness = run(func_example, dimensionality, bounds_example, maximum_evaluations)
print("Best Fitness Found:", best_fitness)
