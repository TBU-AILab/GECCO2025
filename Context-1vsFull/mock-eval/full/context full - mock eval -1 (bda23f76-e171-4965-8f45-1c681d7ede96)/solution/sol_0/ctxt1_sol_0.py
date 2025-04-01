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

def evaluate_population(population, func):
    fitness_values = [func(individual) for individual in population]
    return fitness_values

def run(func, dim, bounds, max_evals):
    pop_size = 30
    w = 0.5
    c1 = 1.5
    c2 = 1.5
    velocities = [[random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds] for _ in range(pop_size)]

    population = initialize_population(pop_size, dim, bounds)
    fitness_values = evaluate_population(population, func)
    global_best_value = min(fitness_values)
    global_best_position = population[fitness_values.index(global_best_value)]
    local_best_positions = population[:]
    local_best_values = fitness_values[:]

    evals = pop_size

    while evals < max_evals:
        for i, individual in enumerate(population):
            velocities[i] = [w * vel +
                             c1 * random.random() * (l_best - curr) +
                             c2 * random.random() * (g_best - curr)
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

    return global_best_value
