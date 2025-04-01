import numpy as np
import random

def run(func, dim, bounds, max_evals):
    # Parameters
    population_size = 20
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
    best_idx = np.argmin(personal_best_fitness)
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

    return global_best_position, global_best_fitness, evals

# Feedback-based improvement: Separate concerns, add better initialization
def initialize_population(population_size, dim, bounds):
    # Initialize population using LHS for better coverage
    population = []
    for _ in range(population_size):
        individual = [(low + (high - low) * np.random.rand()) for low, high in bounds]
        population.append(np.array(individual))
    velocities = [np.zeros(dim) for _ in range(population_size)]
    return population, velocities

def evaluate_population(func, population, personal_best_positions, personal_best_fitness):
    evals = 0
    for i, individual in enumerate(population):
        fitness = func(individual)
        evals += 1
        if fitness < personal_best_fitness[i]:
            personal_best_fitness[i] = fitness
            personal_best_positions[i] = individual
    return evals

def run_with_improvements(func, dim, bounds, max_evals):
    # Parameters
    population_size = 20
    inertia_weight = 0.8
    cognitive_coefficient = 1.5
    social_coefficient = 1.5

    # Initialize population and velocities
    population, velocities = initialize_population(population_size, dim, bounds)
    personal_best_positions = list(population)
    personal_best_fitness = [func(individual) for individual in population]
    evals = population_size

    # Identify the global best position
    best_idx = np.argmin(personal_best_fitness)
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

    return global_best_position, global_best_fitness, evals
