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

# Initialize optimizer
optimizer = AdaptiveSearchOptimizer(sphere_function, dim=10, bounds=bounds, max_evals=1000)

# Perform optimization
best_value, best_solution = optimizer.optimize()

print(f"Best value found: {best_value}")
print(f"Best solution found: {best_solution}")
#
#This updated optimizer class `AdaptiveSearchOptimizer` has several improvements aimed at enhancing performance based on the provided feedback:
#
#1. **Encapsulation in a Class**: The optimizer is now encapsulated in a class, making it modular and reusable for various functions and configurations.
#
#2. **Evaluation Counting**: The `evaluate()` method ensures accurate counting of function evaluations every time `self.func()` is called.
#
#3. **Dynamic Step Size**: The step size is adapted dynamically both when an improvement is found and when exploration is required. It introduces a minimum size for the exploratory capability.
#
#4. **Adaptive Parameters**: The update rule for step sizes and its bounds account for both exploiting local improvements and exploring new areas efficiently.
#
#5. **Initial Exploration Factor**: Initializes step sizes based on exploration needs defined by the differences (limits) of bounds, scaled with a configurable `exploration_factor`.
#
#6. **Control Structure**: Logic for adjusting step size ensures a continuous balance between improving the current solution and searching broader parameter areas.
#
#These changes aim to achieve better performance when faced with challenging optimization problems.
