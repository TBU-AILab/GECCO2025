import random

def pso(func, dim, bounds, max_evals):
    best_value = float('inf')
    best_position = None
    evals = 0

    # PSO parameters
    population_size = 20
    w = 0.5  # inertia weight
    c1 = 1.5  # cognitive (personal) weight
    c2 = 1.5  # social (global) weight
    min_velocity = -1.0
    max_velocity = 1.0

    # Initialize a population of solutions and velocities randomly
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    velocities = [[random.uniform(min_velocity, max_velocity) for _ in range(dim)] for _ in range(population_size)]

    # Initialize personal bests and global best
    pbest_positions = population[:]
    pbest_values = [func(individual) for individual in population]
    evals += population_size

    gbest_value = min(pbest_values)
    gbest_position = pbest_positions[pbest_values.index(gbest_value)]

    while evals < max_evals:
        for i in range(population_size):
            # Update velocity and position of each particle
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()

                velocities[i][d] = (
                    w * velocities[i][d]
                    + c1 * r1 * (pbest_positions[i][d] - population[i][d])
                    + c2 * r2 * (gbest_position[d] - population[i][d])
                )
                velocities[i][d] = max(min(velocities[i][d], max_velocity), min_velocity)

                population[i][d] += velocities[i][d]
                population[i][d] = max(min(population[i][d], bounds[d][1]), bounds[d][0])

            # Evaluate the fitness
            fitness = func(population[i])
            evals += 1

            # Update personal best
            if fitness < pbest_values[i]:
                pbest_values[i] = fitness
                pbest_positions[i] = population[i][:]

            # Update global best
            if fitness < gbest_value:
                gbest_value = fitness
                gbest_position = population[i][:]

            # Early stopping condition if max_evals is reached
            if evals >= max_evals:
                break

    return gbest_value, gbest_position

# Example usage
def sample_function(x):
    return sum(xi ** 2 for xi in x)

# Dimensions and bounds for the problem
dimensions = 5
bounds = [(-10, 10) for _ in range(dimensions)]
maximum_evaluations = 1000

# Running the PSO algorithm
best_value, best_position = pso(sample_function, dimensions, bounds, maximum_evaluations)
print(f"Best Value: {best_value}\nBest Position: {best_position}")
#This implementation adds velocity clamping to ensure particles do not move too quickly through the search space, improving the algorithm's stability and convergence rates. The inertia weight is reduced to balance exploration and exploitation better. Additionally, it includes an example usage at the end to demonstrate how the function can be called, adhering to the latest Python practices.
