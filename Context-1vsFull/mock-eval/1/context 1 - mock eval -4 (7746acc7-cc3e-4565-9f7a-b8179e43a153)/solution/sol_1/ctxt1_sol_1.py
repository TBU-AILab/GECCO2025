import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    best_position = None
    evals = 0

    # Initialize a population of solutions randomly
    population_size = 20
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)] for _ in range(population_size)]
    velocities = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(population_size)]

    # Initialize personal bests and the global best
    pbest_positions = population[:]
    pbest_values = [func(individual) for individual in population]
    evals += population_size

    # Identify the current global best solution
    gbest_value = min(pbest_values)
    gbest_position = pbest_positions[pbest_values.index(gbest_value)]

    # PSO parameters
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive (personal) weight
    c2 = 1.5  # social (global) weight

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
                population[i][d] += velocities[i][d]

                # Ensure position is within bounds
                if population[i][d] < bounds[d][0]:
                    population[i][d] = bounds[d][0]
                elif population[i][d] > bounds[d][1]:
                    population[i][d] = bounds[d][1]

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

            # Stop if max_evals is reached
            if evals >= max_evals:
                break

    return gbest_value
#This improved implementation maintains all key features of a standard Particle Swarm Optimization (PSO) algorithm, with adjustments for balancing exploration and exploitation through PSO parameters. It ensures that the particles' positions remain within defined bounds and efficiently updates both personal and global bests with every iteration.
