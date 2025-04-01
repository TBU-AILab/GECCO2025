import random

# Function to encapsulate the full metaheuristic search
def run(func, dim, bounds, max_evals):
    # Initialization of variables for search process
    population_size = 20
    perturbation_strength = 0.1

    # Create an initial population of solutions within provided bounds
    population = [
        ([low + (high - low) * random.random() for low, high in bounds], None) 
        for _ in range(population_size)
    ]
    
    evals = 0

    # Evaluate the initial population
    for i in range(population_size):
        individual = population[i][0]
        population[i] = (individual, func(individual))
        evals += 1  # Count each function evaluation
    
    best_fitness = min(population, key=lambda x: x[1])[1]
    
    while evals < max_evals:
        offspring = []
        
        for individual, fitness in population:
            # Create a new solution by perturbing the current one
            new_solution = [
                max(bounds[d][0], min(bounds[d][1], individual[d] + perturbation_strength * (random.random() * 2 - 1)))
                for d in range(dim)
            ]
            new_fitness = func(new_solution)
            evals += 1  # Count each function evaluation
            
            # Select the better solution
            if new_fitness < fitness:
                offspring.append((new_solution, new_fitness))
            else:
                offspring.append((individual, fitness))
        
        # Sort the offspring and select the top solutions to continue
        population = sorted(offspring, key=lambda x: x[1])[:population_size]
        best_fitness = min(population, key=lambda x: x[1])[1]
    
    return best_fitness

# Example usage assuming an external func
def func_example(x):
    """Example function to evaluate; just the sum of squares of the solution vector."""
    return sum(x_i**2 for x_i in x)

# Define the problem dimensionality and bounds of the search space
dimensionality = 5
bounds_example = [(-5, 5) for _ in range(dimensionality)]
maximum_evaluations = 100

# Run the search and output the best fitness value found
best_fitness = run(func_example, dimensionality, bounds_example, maximum_evaluations)
print("Best Fitness Found:", best_fitness)
