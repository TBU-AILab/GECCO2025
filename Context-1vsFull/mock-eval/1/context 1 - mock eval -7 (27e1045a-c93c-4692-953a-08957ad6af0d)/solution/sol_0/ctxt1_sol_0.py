import numpy as np
import random

def run(func, dim, bounds, max_evals):
    # Initialize the best solution to random values within the bounds
    best_position = np.array([random.uniform(low, high) for low, high in bounds])
    best_fitness = func(best_position)
    evals = 1

    # Define exploration parameters
    population_size = 20
    exploration_factor = 0.7  # Exploration weight for new particles

    # Initialize the population with random positions
    population = [np.array([random.uniform(low, high) for low, high in bounds]) for _ in range(population_size)]
    population_fitness = [func(individual) for individual in population]
    evals += population_size

    # Update the best found position
    for i in range(population_size):
        if population_fitness[i] < best_fitness:
            best_fitness = population_fitness[i]
            best_position = population[i]

    while evals < max_evals:
        # Velocity and position update for each particle
        for i in range(population_size):
            # Generate a random particle around the current position for exploration
            exploration_vector = np.array([random.uniform(-1, 1) for _ in range(dim)]) * exploration_factor
            new_position = population[i] + exploration_vector

            # Ensure new position is within bounds
            new_position = np.clip(new_position, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate the new position
            new_fitness = func(new_position)
            evals += 1
            if evals >= max_evals:
                break

            # Update the population with new position if it is better
            if new_fitness < population_fitness[i]:
                population[i] = new_position
                population_fitness[i] = new_fitness

            # Update the global best if the new fitness is the best found so far
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_position = new_position

    return best_fitness
