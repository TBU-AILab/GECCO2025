import random

class ConstrainedOptimizer:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.best_solution = None
        self.best_fitness = float('inf')

    def evaluate(self, solution):
        """Evaluate the solution using the provided objective function and count the evaluation."""
        if self.evals >= self.max_evals:
            raise RuntimeError("Maximum number of evaluations exceeded")
        fitness = self.func(solution)
        self.evals += 1
        return fitness

    def initialize_solution(self):
        """Generate a random initial solution within the given bounds."""
        return [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]

    def neighborhood_search(self, point, scale=0.1):
        """Conducts a neighborhood search around the current point."""
        return [
            max(self.bounds[i][0], min(self.bounds[i][1], point[i] + random.uniform(-scale, scale)))
            for i in range(self.dim)
        ]

    def run(self):
        """Run the optimization routine using a simple neighborhood search."""
        current_solution = self.initialize_solution()
        current_fitness = self.evaluate(current_solution)

        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = current_solution

        while self.evals < self.max_evals:
            scale = 0.1 * (1 - self.evals / self.max_evals)

            new_solution = self.neighborhood_search(current_solution, scale)
            new_fitness = self.evaluate(new_solution)

            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness

                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_solution

        return self.best_fitness, self.best_solution

# Example usage
def example_function(x):
    return sum(xi**2 for xi in x)  # Simple sphere function for testing

if __name__ == "__main__":
    dim = 5
    bounds = [(-5, 5) for _ in range(dim)]
    max_evals = 1000

    optimizer = ConstrainedOptimizer(example_function, dim, bounds, max_evals)
    try:
        best_fitness, best_solution = optimizer.run()
        print("Best Fitness:", best_fitness)
        print("Best Solution:", best_solution)
    except RuntimeError as e:
        print(e)
