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


def evaluate_population(population, func):
    fitness_values = [func(individual) for individual in population]
    return fitness_values


def run(func, dim, bounds, max_evals):
    pop_size = 50  # Larger population size for diversity
    inertia_init = 0.9  # Initial inertia
    inertia_final = 0.4  # Final inertia
    cognitive = 2.05  # Cognitive acceleration coefficient
    social = 2.05  # Social acceleration coefficient
    velocities = [[random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds] for _ in range(pop_size)]

    population = initialize_population(pop_size, dim, bounds)
    fitness_values = evaluate_population(population, func)
    evals = len(population)

    # Initialize bests
    global_best_value = min(fitness_values)
    global_best_position = population[fitness_values.index(global_best_value)]
    local_best_positions = population[:]
    local_best_values = fitness_values[:]

    while evals < max_evals:
        inertia = inertia_init - ((inertia_init - inertia_final) * (evals / max_evals))  # Decrease inertia over time

        for i, individual in enumerate(population):
            velocities[i] = [inertia * vel +
                             cognitive * random.random() * (l_best - curr) +
                             social * random.random() * (g_best - curr)
                             for vel, curr, l_best, g_best in zip(velocities[i], individual, local_best_positions[i], global_best_position)]

            population[i] = update_position(individual, velocities[i], bounds)
            fitness = func(population[i])
            evals += 1

            # Update local best
            if fitness < local_best_values[i]:
                local_best_values[i] = fitness
                local_best_positions[i] = population[i][:]

            # Update global best
            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = population[i][:]

            if evals >= max_evals:
                break

    return global_best_value
