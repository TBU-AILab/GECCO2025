import random

def run(func, dim, bounds, max_evals):
    def clip(value, low, high):
        """Ensure the value is within the provided bounds."""
        return max(min(value, high), low)

    def mutate(individual, mutation_strength=0.1):
        """Mutate an individual by adding a small random change."""
        return [
            clip(
                individual[i] + random.uniform(-mutation_strength, mutation_strength),
                bounds[i][0],
                bounds[i][1],
            )
            for i in range(dim)
        ]

    def crossover(parent1, parent2):
        """Perform crossover by averaging two parents."""
        return [
            (parent1[i] + parent2[i]) / 2.0
            for i in range(dim)
        ]

    population_size = 20
    population = [
        [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
        for _ in range(population_size)
    ]

    fitness_scores = [func(ind) for ind in population]
    evals = len(population)
    best_idx = fitness_scores.index(min(fitness_scores))
    best = population[best_idx]
    best_fitness = fitness_scores[best_idx]

    while evals < max_evals:
        # Select parents: tournament selection
        new_population = []
        tournament_size = max(2, population_size // 5)
        
        for _ in range(population_size):
            tournament = random.sample(population, tournament_size)
            tournament_fitness = [func(ind) for ind in tournament]
            evals += tournament_size
            winner_idx = tournament_fitness.index(min(tournament_fitness))
            best_of_tournament = tournament[winner_idx]
            new_population.append(best_of_tournament)
        
        # Create next generation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring)
            next_generation.append(offspring)

        population = next_generation
        fitness_scores = [func(ind) for ind in population]
        evals += len(population)
        current_best_idx = fitness_scores.index(min(fitness_scores))
        
        if fitness_scores[current_best_idx] < best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best = population[current_best_idx]

    return best_fitness
#
#In this revised algorithm:
#
#- **Tournament Selection** is used to allow a more competitive selection pressure, preventing premature convergence by randomly selecting a group of solutions and picking the best one.
#- The evaluation count `evals` is correctly updated every time the objective function `func()` is called.
#- Mutation strength and crossover operations are clearly defined, with individuals being clipped to remain within bounds.
#- Code comments have been added to enhance clarity and maintainability.
#- `best` now correctly tracks the best solution found, ensuring that the minimum fitness value is obtained over the search.
