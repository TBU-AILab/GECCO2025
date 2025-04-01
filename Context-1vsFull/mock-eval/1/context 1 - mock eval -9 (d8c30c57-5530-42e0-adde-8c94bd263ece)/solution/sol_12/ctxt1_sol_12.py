import random

def main():
    # Example quadratic function to minimize
    def example_cost_function(x):
        return sum(x_i**2 for x_i in x)

    # Define the problem dimensionality and bounds
    dimensionality = 5
    bounds_example = [(-5, 5) for _ in range(dimensionality)]
    maximum_evaluations = 100
    
    # Run the optimizer and print the best fitness found
    best_fitness = run_optimizer(example_cost_function, dimensionality, bounds_example, maximum_evaluations)
    print("Best Fitness Found:", best_fitness)

def run_optimizer(func, dim, bounds, max_evals):
    population_size = 20
    perturbation_strength = 0.1

    # Initialize a population with random solutions
    population = [
        ([random.uniform(low, high) for low, high in bounds], None)
        for _ in range(population_size)
    ]
    
    evals = 0

    # Evaluate the initial population
    for i in range(population_size):
        individual = population[i][0]
        population[i] = (individual, func(individual))
        evals += 1
    
    best_fitness = min(population, key=lambda x: x[1])[1]
    
    # Main loop of the optimizer
    while evals < max_evals:
        offspring = []
        
        for individual, fitness in population:
            # Create a new solution by perturbing the current individual
            new_solution = [
                max(bounds[d][0], min(bounds[d][1], individual[d] + perturbation_strength * (random.random() * 2 - 1)))
                for d in range(dim)
            ]
            new_fitness = func(new_solution)
            evals += 1
            
            # Select the best solution between the current one and the perturbed one
            if new_fitness < fitness:
                offspring.append((new_solution, new_fitness))
            else:
                offspring.append((individual, fitness))
        
        # Select the best individuals to form the new population
        population = sorted(offspring, key=lambda x: x[1])[:population_size]
        best_fitness = min(population, key=lambda x: x[1])[1]
    
    return best_fitness

if __name__ == "__main__":
    main()
