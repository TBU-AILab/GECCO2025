import random

class MetaheuristicOptimizer:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals_count = 0

    def clip(self, value, low, high):
        """Ensure the value is within the provided bounds."""
        return max(min(value, high), low)

    def evaluate(self, individual):
        """Evaluate an individual and increment evaluations."""
        self.evals_count += 1
        return self.func(individual)

    def mutate(self, individual, mutation_strength=0.1):
        """Mutate an individual by adding a small random change."""
        return [
            self.clip(
                individual[i] + random.uniform(-mutation_strength, mutation_strength),
                self.bounds[i][0],
                self.bounds[i][1],
            )
            for i in range(self.dim)
        ]

    def crossover(self, parent1, parent2):
        """Perform crossover by averaging two parents."""
        return [
            (parent1[i] + parent2[i]) / 2.0
            for i in range(self.dim)
        ]

    def optimize(self):
        population_size = 20
        population = [
            [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
            for _ in range(population_size)
        ]

        fitness_scores = [self.evaluate(ind) for ind in population]
        best_idx = fitness_scores.index(min(fitness_scores))
        best = population[best_idx]
        best_fitness = fitness_scores[best_idx]

        while self.evals_count < self.max_evals:
            new_population = []
            tournament_size = max(2, population_size // 5)

            for _ in range(population_size):
                tournament = random.sample(population, tournament_size)
                tournament_fitness = [self.evaluate(ind) for ind in tournament]
                winner_idx = tournament_fitness.index(min(tournament_fitness))
                best_of_tournament = tournament[winner_idx]
                new_population.append(best_of_tournament)

            next_generation = []
            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(new_population, 2)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                next_generation.append(offspring)

            population = next_generation
            fitness_scores = [self.evaluate(ind) for ind in population]
            current_best_idx = fitness_scores.index(min(fitness_scores))

            if fitness_scores[current_best_idx] < best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best = population[current_best_idx]

        return best, best_fitness

def run(func, dim, bounds, max_evals):
    optimizer = MetaheuristicOptimizer(func, dim, bounds, max_evals)
    best_solution, best_fitness = optimizer.optimize()
    print(f"Best fitness: {best_fitness}")
    return best_solution

# Example usage:
# def obj_function(x):
#     return sum(xi ** 2 for xi in x)
# best = run(obj_function, dim=5, bounds=[(-10, 10)] * 5, max_evals=500)
#
#In this enhanced version:
#
#- A class `MetaheuristicOptimizer` encapsulates our optimization logic, which helps in managing state and organizing code more effectively.
#- The `evaluate` method properly counts evaluations and is called every time we need to compute the objective function.
#- An object-oriented approach allows better extensibility and maintainability.
#- The input function and parameters are provided at the instantiation of the `MetaheuristicOptimizer`, promoting cleaner interaction and separation of concerns.
