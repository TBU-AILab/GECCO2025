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
        """Mutate an individual by adding a random change based on mutation strength."""
        return [
            self.clip(
                individual[i] + random.gauss(0, self.mutation_strength),
                self.bounds[i][0], self.bounds[i][1]
            ) for i in range(self.dim)
        ]

    def crossover(self, parent1, parent2):
        """Perform crossover by interpolating between parents."""
        return [
            self.clip(
                random.uniform(0, 1) * parent1[i] + (1 - random.uniform(0, 1)) * parent2[i],
                self.bounds[i][0], self.bounds[i][1]
            ) for i in range(self.dim)
        ]

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
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)

            population = new_population
            fitness_scores = [self.evaluate(ind) for ind in population]
            current_best_idx = min(range(self.population_size), key=lambda idx: fitness_scores[idx])

            # Improve convergence by adjusting mutation strength over time
            self.mutation_strength *= 0.99

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
#### Key Improvement:
#
#1. **Dynamic Mutation Strength**: The mutation strength now gradually decreases over time by multiplying it by 0.99 in each iteration. This adjustment allows the optimizer to start with larger exploratory steps and progressively refine its search as it nears convergence. This balance between exploration and exploitation can enhance the solution's quality and help avoid premature convergence.
