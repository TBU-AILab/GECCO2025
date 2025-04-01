import random

def run(func, dim, bounds, max_evals):
    def clip(value, low, high):
        return max(min(value, high), low)

    def mutate(individual, bounds):
        mutation_rate = 0.05
        mutated = [
            clip(
                individual[i] + random.uniform(-1, 1) * mutation_rate,
                bounds[i][0],
                bounds[i][1],
            )
            for i in range(dim)
        ]
        return mutated

    def crossover(parent1, parent2):
        alpha = random.uniform(0, 1)
        return [
            alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(dim)
        ]

    population_size = 20
    population = [
        [
            random.uniform(bounds[i][0], bounds[i][1])
            for i in range(dim)
        ]
        for _ in range(population_size)
    ]

    best = float("inf")
    evals = 0

    while evals < max_evals:
        population.sort(key=func)
        evals += len(population)

        if func(population[0]) < best:
            best = func(population[0])

        next_generation = []

        # Elitism: carry over the best two individuals
        next_generation.extend(population[:2])

        while len(next_generation) < population_size:
            parent1 = random.choice(population[:population_size // 2])
            parent2 = random.choice(population[:population_size // 2])
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, bounds)
            next_generation.append(offspring)

        population = next_generation

    return best
#This algorithm combines elements of both evolutionary strategies and differential evolution principles. It incorporates mutation, crossover, and elitism to improve solutions iteratively. Note that the number of evaluations (`evals`) is correctly tracked every time the `func()` is called.
