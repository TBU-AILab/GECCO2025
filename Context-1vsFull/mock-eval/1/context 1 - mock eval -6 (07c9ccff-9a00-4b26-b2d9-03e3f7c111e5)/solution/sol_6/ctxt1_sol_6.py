import random

class MetaheuristicOptimizer:
    def __init__(self, func, dim, bounds, max_evals, population_size=20, mutation_strength=0.1):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.population_size = population_size
        self.mutation_strength = mutation_strength
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

    def mutate(self, individual):
        """Mutate an individual by adding a small random change."""
        return [
            self.clip(
                individual[i] + random.uniform(-self.mutation_strength, self.mutation_strength),
                self.bounds[i][0], self.bounds[i][1]
            ) for i in range(self.dim)
        ]

    def crossover(self, parent1, parent2):
        """Perform crossover by averaging two parents."""
        offspring = [
            self.clip(
                (parent1[i] + parent2[i]) / 2.0, self.bounds[i][0], self.bounds[i][1]
            ) for i in range(self.dim)
        ]
        return offspring

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select the best individual from a tournament."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda idx: fitness_scores[idx])
        return population[best_idx]

    def optimize(self):
        population = [
            [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
            for _ in range(self.population_size)
        ]

        fitness_scores = [self.evaluate(ind) for ind in population]
        best_idx = min(range(self.population_size), key=lambda idx: fitness_scores[idx])
        best, best_fitness = population[best_idx][:], fitness_scores[best_idx]

        while self.evals_count < self.max_evals:
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                while parent1 == parent2:
                    parent2 = self.tournament_selection(population, fitness_scores)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)

            population = new_population
            fitness_scores = [self.evaluate(ind) for ind in population]
            current_best_idx = min(range(self.population_size), key=lambda idx: fitness_scores[idx])

            if fitness_scores[current_best_idx] < best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best = population[current_best_idx][:]

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
#### Key Enhancements:
#
#1. **Variable Naming Consistency**: Ensured consistent and meaningful naming, enhancing readability.
#
#2. **Encapsulation of Logic**: Crossover and mutation logic is encapsulated and streamlined.
#
#3. **Efficient Parent Selection**: Ensured uniqueness of parents using a simple while loop within the optimization process without redundancy.
#
#4. **Improved Code Efficiency**: Removed unnecessary deep copy in crossover by directly clipping results.
#
#These updates should make the optimizer clearer to read and understand, with improved logical flow and performance.
