import numpy as np
import random

def run(func, dim, bounds, max_evals):
    # Parameters
    population_size = 20
    exploration_factor = 0.5
    inertia_weight = 0.8
    cognitive_coefficient = 1.5
    social_coefficient = 1.5

    # Initialize population and velocities
    population = [np.array([random.uniform(low, high) for low, high in bounds]) for _ in range(population_size)]
    velocities = [np.zeros(dim) for _ in range(population_size)]
    personal_best_positions = list(population)
    personal_best_fitness = [func(individual) for individual in population]
    evals = population_size

    # Identify the global best position
    best_idx = personal_best_fitness.index(min(personal_best_fitness))
    global_best_fitness = personal_best_fitness[best_idx]
    global_best_position = personal_best_positions[best_idx]

    while evals < max_evals:
        for i in range(population_size):
            # Update velocity
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (inertia_weight * velocities[i] + 
                             cognitive_coefficient * r1 * (personal_best_positions[i] - population[i]) +
                             social_coefficient * r2 * (global_best_position - population[i]))

            # Update the position
            population[i] = population[i] + velocities[i]
            population[i] = np.clip(population[i], [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate new fitness
            fitness = func(population[i])
            evals += 1
            if evals >= max_evals:
                break

            # Update personal bests
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = population[i]

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = population[i]

    return global_best_fitness
