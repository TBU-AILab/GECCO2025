import numpy as np

def func(x):
    # Dummy objective function for demonstration
    return sum(x**2)

def initialize_population(population_size, dim, bounds):
    # Initializing population within bounds
    population = []
    lower_bounds, upper_bounds = np.array(bounds).T
    for _ in range(population_size):
        individual = np.random.uniform(lower_bounds, upper_bounds)
        population.append(individual)
    velocities = [np.zeros(dim) for _ in range(population_size)]
    return np.array(population), np.array(velocities)

def evaluate_population(func, population):
    return np.array([func(individual) for individual in population])

def update_personal_best(population, fitness, personal_best_positions, personal_best_fitness):
    mask = fitness < personal_best_fitness
    personal_best_positions[mask] = population[mask]
    personal_best_fitness[mask] = fitness[mask]
    return personal_best_positions, personal_best_fitness

def find_global_best(personal_best_positions, personal_best_fitness):
    best_idx = np.argmin(personal_best_fitness)
    return personal_best_positions[best_idx], personal_best_fitness[best_idx]

def run(func, dim, bounds, max_evals):
    # Parameters
    population_size = 20
    inertia_weight = 0.7
    cognitive_coefficient = 1.5
    social_coefficient = 1.5

    # Random seed for reproducibility
    np.random.seed(42)

    # Initialize population and velocities
    population, velocities = initialize_population(population_size, dim, bounds)
    personal_best_positions = np.copy(population)
    personal_best_fitness = evaluate_population(func, population)
    evals = population_size

    # Identify global best position
    global_best_position, global_best_fitness = find_global_best(personal_best_positions, personal_best_fitness)

    while evals < max_evals:
        for i in range(population_size):
            if evals >= max_evals:
                break

            # Update velocity
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (inertia_weight * velocities[i] + 
                             cognitive_coefficient * r1 * (personal_best_positions[i] - population[i]) +
                             social_coefficient * r2 * (global_best_position - population[i]))

            # Update position
            population[i] = population[i] + velocities[i]
            population[i] = np.clip(population[i], [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate fitness
            fitness = func(population[i])
            evals += 1

            # Update personal bests
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = population[i]

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = population[i]

    return global_best_position, global_best_fitness, evals

# Example usage
if __name__ == '__main__':
    dim = 5
    bounds = [(-5, 5)] * dim
    max_evals = 1000
    best_position, best_fitness, total_evals = run(func, dim, bounds, max_evals)
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")
    print(f"Total Func Evaluations: {total_evals}")
