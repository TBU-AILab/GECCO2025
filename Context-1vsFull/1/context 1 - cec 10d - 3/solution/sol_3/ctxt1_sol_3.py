import random

class AdaptiveSearchOptimizer:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0

    def evaluate(self, solution):
        self.evals += 1
        return self.func(solution)

    def optimize(self):
        # Initialize the best solution and step size dynamically
        current_solution = [random.uniform(low, high) for low, high in self.bounds]
        current_fitness = self.evaluate(current_solution)
        best_solution = current_solution[:]
        best_fitness = current_fitness

        # Define adaptive parameters
        adaptive_rate = 0.9
        exploration_factor = 0.1
        min_step_size = 0.001

        # Dynamic step size initialization
        step_size = [(high - low) * exploration_factor for low, high in self.bounds]

        while self.evals < self.max_evals:
            # Generate and evaluate new candidate with an adaptive Gaussian perturbation
            candidate_solution = [
                max(min(best_solution[i] + random.gauss(0, step_size[i] / 3.0), self.bounds[i][1]), self.bounds[i][0])
                for i in range(self.dim)
            ]
            candidate_fitness = self.evaluate(candidate_solution)

            # Update adaptively based on performance
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution[:]
                best_fitness = candidate_fitness
                step_size = [max(step * adaptive_rate, min_step_size * (high - low)) for step, (low, high) in zip(step_size, self.bounds)]
            else:
                step_size = [step / adaptive_rate for step in step_size]

        return best_fitness, best_solution

# Example function to optimize (e.g., Sphere function)
def sphere_function(x):
    return sum(xi ** 2 for xi in x)

# Boundaries for each dimension
bounds = [(-100, 100) for _ in range(10)]

# Helper function to run the optimizer
def run_optimizer():
    # Initialize optimizer
    optimizer = AdaptiveSearchOptimizer(sphere_function, dim=10, bounds=bounds, max_evals=1000)

    # Perform optimization
    best_value, best_solution = optimizer.optimize()

    print(f"Best value found: {best_value}")
    print(f"Best solution found: {best_solution}")

# Run the optimizer
if __name__ == "__main__":
    run_optimizer()
#
#I've made these crucial changes:
#
#1. **Import Correction**: Ensured `random` is properly imported before use, resolving the `NameError`.
#
#2. **Encapsulation of Execution**: Wrapped the optimizer execution into a `run_optimizer()` function and called it within the `if __name__ == "__main__":` block to fix the `KeyError` related to `run`.
#
#This code can now be executed with properly encapsulated logic, making it flexible for further testing and modification.
