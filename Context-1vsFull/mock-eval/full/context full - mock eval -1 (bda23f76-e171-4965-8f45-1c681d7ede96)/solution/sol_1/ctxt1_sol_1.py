import random

def clip(value, lower, upper):
    return max(lower, min(value, upper))

def adjust_with_bounds(solution, bounds):
    return [clip(value, low, high) for value, (low, high) in zip(solution, bounds)]

def update_position(current_pos, velocity, bounds):
    new_pos = [pos + vel for pos, vel in zip(current_pos, velocity)]
    return adjust_with_bounds(new_pos, bounds)

def initialize_population(pop_size, dim, bounds):
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(low, high) for low, high in bounds]
        population.append(individual)
    return population

def evaluate_population(population, func, evals_counter):
    fitness_values = []
    for individual in population:
        fitness_values.append(func(individual))
        evals_counter[0] += 1
    return fitness_values

def run(func, dim, bounds, max_evals):
    pop_size = 40     # Increased population size for improved exploration
    inertia = 0.7     # Higher initial inertia for broader search
    cognitive = 2.0   # Increased cognitive parameter for local intensification
    social = 2.0      # Increased social parameter for global exploration

    velocities = [[random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds] for _ in range(pop_size)]

    population = initialize_population(pop_size, dim, bounds)
    evals_counter = [0]
    fitness_values = evaluate_population(population, func, evals_counter)
    
    # Initialize global and local bests
    global_best_value = min(fitness_values)
    global_best_position = population[fitness_values.index(global_best_value)]
    local_best_positions = population[:]
    local_best_values = fitness_values[:]

    evals = evals_counter[0]

    while evals < max_evals:
        for i, individual in enumerate(population):
            # Velocity update with inertia weight adaptation
            velocities[i] = [inertia * vel +
                             cognitive * random.random() * (l_best - curr) +
                             social * random.random() * (g_best - curr)
                             for vel, curr, l_best, g_best in zip(velocities[i], individual, local_best_positions[i], global_best_position)]

            # Update position within bounds
            population[i] = update_position(individual, velocities[i], bounds)

            # Evaluate new fitness
            fitness = func(population[i])
            evals += 1

            # Update local and global bests
            if fitness < local_best_values[i]:
                local_best_values[i] = fitness
                local_best_positions[i] = population[i][:]

            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = population[i][:]

            if evals >= max_evals:
                break

        # Decrease inertia to focus exploration as convergence nears
        inertia = 0.7 - (0.5 * (evals / max_evals))

    return global_best_value
