import random

class Evaluator:
    def __init__(self, func):
        self.func = func
        self.evals = 0
    
    def evaluate(self, solution):
        self.evals += 1
        return self.func(solution)


class MetaheuristicOptimizer:
    def __init__(self, evaluator, dimensionality, bounds, max_evals):
        self.evaluator = evaluator
        self.dimensionality = dimensionality
        self.bounds = bounds
        self.max_evals = max_evals

    def random_solution(self):
        return [random.uniform(low, high) for low, high in self.bounds]

    def perturb_solution(self, solution, strength):
        return [
            max(self.bounds[d][0], min(self.bounds[d][1], solution[d] + strength * (random.random() * 2 - 1)))
            for d in range(self.dimensionality)
        ]

    def select_best(self, population):
        return min(population, key=lambda x: x[1])

    def optimize(self):
        best_solution = self.random_solution()
        best_fitness = self.evaluator.evaluate(best_solution)

        while self.evaluator.evals < self.max_evals:
            new_solution = self.perturb_solution(best_solution, 0.1)
            new_fitness = self.evaluator.evaluate(new_solution)

            if new_fitness < best_fitness:
                best_solution, best_fitness = new_solution, new_fitness

        return best_solution, best_fitness


def example_cost_function(x):
    """Example objective function to be minimized."""
    return sum(x_i**2 for x_i in x)

def run(func, dimensionality, bounds, max_evaluations):
    evaluator = Evaluator(func)
    optimizer = MetaheuristicOptimizer(evaluator, dimensionality, bounds, max_evaluations)
    best_solution, best_fitness = optimizer.optimize()
    print("Best Solution Found:", best_solution)
    print("Best Fitness Found:", best_fitness)
    print("Total Function Evaluations:", evaluator.evals)


if __name__ == "__main__":
    dimensionality = 5
    bounds = [(-5, 5) for _ in range(dimensionality)]
    max_evaluations = 100
    
    run(example_cost_function, dimensionality, bounds, max_evaluations)
