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
    return [func(individual) for individual in population]

def run(func, dim, bounds, max_evals):
    pop_size = 50
    inertia_start = 0.9
    inertia_end = 0.4
    cognitive = 2.0
    social = 2.0

    velocities = [[random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds] for _ in range(pop_size)]
    population = initialize_population(pop_size, dim, bounds)
    fitness_values = evaluate_population(population, func)
    evals = len(population)

    global_best_value = min(fitness_values)
    global_best_position = population[fitness_values.index(global_best_value)]
    local_best_positions = population[:]
    local_best_values = fitness_values[:]

    def adaptive_inertia(evaluations):
        return inertia_start - (evaluations / max_evals) * (inertia_start - inertia_end)

    while evals < max_evals:
        inertia = adaptive_inertia(evals)

        for i in range(pop_size):
            velocities[i] = [
                inertia * velocities[i][j] +
                cognitive * random.random() * (local_best_positions[i][j] - population[i][j]) +
                social * random.random() * (global_best_position[j] - population[i][j])
                for j in range(dim)
            ]

            population[i] = update_position(population[i], velocities[i], bounds)
            fitness = func(population[i])
            evals += 1

            if fitness < local_best_values[i]:
                local_best_positions[i] = population[i][:]
                local_best_values[i] = fitness

            if fitness < global_best_value:
                global_best_position = population[i][:]
                global_best_value = fitness

            if evals >= max_evals:
                break

        # Enhanced perturbation strategy for better exploration
        if evals % (pop_size * 10) == 0:
            for i in range(pop_size):
                if random.random() < 0.2:  # Increased chance for perturbation
                    dim_to_mutate = random.sample(range(dim), dim // 5)  # Mutate 20% of the dimensions
                    for j in dim_to_mutate:
                        mutation_strength = 0.1 * (bounds[j][1] - bounds[j][0])
                        population[i][j] += random.gauss(0, mutation_strength)
                    population[i] = adjust_with_bounds(population[i], bounds)

    return global_best_value
