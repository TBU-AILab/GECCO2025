import random

class GeneticAlgorithm:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.population_size = 20
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_probability = 0.1

    def evaluate(self, solution):
        if self.evals >= self.max_evals:
            raise Exception('Maximum evaluations exceeded')
        self.evals += 1
        return self.func(solution)

    def initialize_population(self):
        return [
            [random.uniform(low, high) for low, high in self.bounds]
            for _ in range(self.population_size)
        ]

    def select_parents(self, population):
        tournament_size = 5
        selected = random.sample(population, tournament_size)
        fitness_selected = [(indv, self.evaluate(indv)) for indv in selected]
        fitness_selected.sort(key=lambda x: x[1])
        return fitness_selected[0][0], fitness_selected[1][0]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.dim - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_probability * (1 + 4 * self.evals / self.max_evals):
                low, high = self.bounds[i]
                individual[i] = random.uniform(low, high)

    def replace_worst(self, population, fitness_values, offspring, fitness):
        max_fit_idx = fitness_values.index(max(fitness_values))
        if fitness < fitness_values[max_fit_idx]:
            population[max_fit_idx] = offspring
            fitness_values[max_fit_idx] = fitness

    def run(self):
        population = self.initialize_population()
        fitness_values = [self.evaluate(ind) for ind in population]

        def update_best():
            nonlocal population, fitness_values
            min_fitness_index = fitness_values.index(min(fitness_values))
            if fitness_values[min_fitness_index] < self.best_fitness:
                self.best_fitness = fitness_values[min_fitness_index]
                self.best_solution = population[min_fitness_index]

        update_best()

        while self.evals < self.max_evals:
            parent1, parent2 = self.select_parents(population)
            offspring1, offspring2 = self.crossover(parent1, parent2)
            
            self.mutate(offspring1)
            self.mutate(offspring2)
            
            fitness_offspring1 = self.evaluate(offspring1)
            fitness_offspring2 = self.evaluate(offspring2)
            
            self.replace_worst(population, fitness_values, offspring1, fitness_offspring1)
            self.replace_worst(population, fitness_values, offspring2, fitness_offspring2)

            update_best()

        return self.best_fitness, self.best_solution

def cost_function(x):
    return sum(x_i ** 2 for x_i in x)

if __name__ == "__main__":
    dim = 5
    bounds = [(-5, 5)] * dim
    max_evals = 1000

    optimizer = GeneticAlgorithm(cost_function, dim, bounds, max_evals)
    best_fitness, best_solution = optimizer.run()
    print("Best fitness found:", best_fitness)
    print("Best solution found:", best_solution)
