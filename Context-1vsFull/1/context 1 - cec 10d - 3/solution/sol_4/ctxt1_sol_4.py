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
        current_solution = [random.uniform(low, high) for low, high in self.bounds]
        current_fitness = self.evaluate(current_solution)
        best_solution = current_solution[:]
        best_fitness = current_fitness

        adaptive_rate = 0.9
        exploration_factor = 0.1
        min_step_size = 0.001

        step_size = [(high - low) * exploration_factor for low, high in self.bounds]

        while self.evals < self.max_evals:
            candidate_solution = [
                max(min(best_solution[i] + random.gauss(0, step_size[i] / 3.0), self.bounds[i][1]), self.bounds[i][0])
                for i in range(self.dim)
            ]
            candidate_fitness = self.evaluate(candidate_solution)

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution[:]
                best_fitness = candidate_fitness
                step_size = [max(step * adaptive_rate, min_step_size * (high - low)) for step, (low, high) in zip(step_size, self.bounds)]
            else:
                step_size = [step / adaptive_rate for step in step_size]

        return best_fitness, best_solution

def sphere_function(x):
    return sum(xi ** 2 for xi in x)

bounds = [(-100, 100) for _ in range(10)]

def main():
    optimizer = AdaptiveSearchOptimizer(sphere_function, dim=10, bounds=bounds, max_evals=1000)
    best_value, best_solution = optimizer.optimize()

    print(f"Best value found: {best_value}")
    print(f"Best solution found: {best_solution}")

if __name__ == "__main__":
    main()
#
#This updated code ensures that all execution logic is properly encapsulated within the `main()` function and called within the `if __name__ == "__main__":` block. This setup should resolve any lingering issues with the `KeyError('run')` by ensuring the function that initiates the run has a clear definition and is correctly executed.
