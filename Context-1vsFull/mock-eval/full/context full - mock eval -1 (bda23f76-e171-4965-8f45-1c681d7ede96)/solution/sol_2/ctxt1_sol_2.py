import random

def clip(value, lower, upper):
    return max(lower, min(value, upper))

def adjust_with_bounds(solution, bounds):
    return [clip(value, low, high) for value, (low, high) in zip(solution, bounds)]

def update_position(current_pos, velocity, bounds):
    new_pos = [pos + vel for pos, vel in zip(current_pos, velocity)]
    return adjust_with_bounds(new_pos, bounds)

def initialize_population(pop_size, dim, bounds):
    return [[random.uniform(low, high) for low, high in bounds] for _ in range(pop_size)]

def evaluate_population(population, func, evals_counter):
    return [func(individual) for individual in population]

def run(func, dim, bounds, max_evals):
    pop_size = 50     # Increased population size for improved exploration
    inertia = 0.9     # Start with a higher inertia for improved global exploration
    cognitive = 1.5   # Balance cognitive influence
    social = 1.5      # Balance social influence

    velocities = [[random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds] for _ in range(pop_size)]
    population = initialize_population(pop_size, dim, bounds)
    evals_counter = [0]
    fitness_values = evaluate_population(population, func, evals_counter)
    
    global_best_value = min(fitness_values)
    global_best_position = population[fitness_values.index(global_best_value)]
    local_best_positions = population[:]
    local_best_values = fitness_values[:]

    evals = len(population)

    while evals < max_evals:
        for i, individual in enumerate(population):
            velocities[i] = [inertia * vel +
                             cognitive * random.random() * (l_best - curr) +
                             social * random.random() * (g_best - curr)
                             for vel, curr, l_best, g_best in zip(velocities[i], individual, local_best_positions[i], global_best_position)]

            population[i] = update_position(individual, velocities[i], bounds)
            fitness = func(population[i])
            evals += 1

            if fitness < local_best_values[i]:
                local_best_values[i] = fitness
                local_best_positions[i] = population[i][:]

            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = population[i][:]

            if evals >= max_evals:
                break

        inertia = 0.9 - (0.8 * (evals / max_evals))  # Linearly decreasing inertia

    return global_best_value
