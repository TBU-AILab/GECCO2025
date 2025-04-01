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
        if self.evals_count >= self.max_evals:
            return float('inf')
        self.evals_count += 1
        return self.func(individual)

    def mutate(self, individual, mutation_strength=0.1):
        """Mutate an individual by adding a small random change."""
        return [
            self.clip(
                individual[i] + random.uniform(-mutation_strength, mutation_strength),
                self.bounds[i][0], self.bounds[i][1]
            ) for i in range(self.dim)
        ]

    def crossover(self, parent1, parent2):
        """Perform crossover by averaging two parents."""
        return [
            self.clip(
                (parent1[i] + parent2[i]) / 2.0, self.bounds[i][0], self.bounds[i][1]
            ) for i in range(self.dim)
        ]

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select the best individual from a tournament."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda idx: fitness_scores[idx])
        return population[best_idx]

    def optimize(self):
        population_size = 20
        population = [
            [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
            for _ in range(population_size)
        ]

        fitness_scores = [self.evaluate(ind) for ind in population]
        best_idx = min(range(population_size), key=lambda idx: fitness_scores[idx])
        best, best_fitness = population[best_idx], fitness_scores[best_idx]

        while self.evals_count < self.max_evals:
            new_population = []
            for _ in range(population_size):
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                if parent1 != parent2:  # Ensure parents are different
                    offspring = self.crossover(parent1, parent2)
                else:
                    offspring = parent1[:]  # If same, just clone
                offspring = self.mutate(offspring)
                new_population.append(offspring)

            population = new_population
            fitness_scores = [self.evaluate(ind) for ind in population]
            current_best_idx = min(range(population_size), key=lambda idx: fitness_scores[idx])

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
#### Key Modifications:
#
#1. **Parent Selection Check**: Added a check during crossover to ensure parent1 and parent2 are not the same. If they are, the offspring simply clones one of the parents.
#
#2. **Direct Fitness Comparison**: Streamlined the selection of the best index and comparison to always use `min` with a key function for better readability and performance.
#
#3. **Maintain Integrity**: The algorithm takes care to maintain an internal state, ensuring no global mutations happen outside the class instance. This ensures encapsulation and easier debugging.
