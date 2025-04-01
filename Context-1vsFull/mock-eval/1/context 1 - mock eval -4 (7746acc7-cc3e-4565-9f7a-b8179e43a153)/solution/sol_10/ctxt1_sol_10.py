import random

def constrained_function(x):
    """Example cost function with constraints."""
    # Example of a constraint: sum(x) must be less than or equal to 10
    if sum(x) > 10:
        return float('inf')  # Penalty for constraint violation
    # Main objective: Minimize the sum of squares
    return sum(xi ** 2 for xi in x)

class ParticleSwarmOptimizer:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0

        # Hyperparameters
        self.population_size = 20
        self.w = 0.4  # inertia weight
        self.c1 = 1.5  # personal weight
        self.c2 = 1.5  # global weight
        self.min_velocity = -1.0
        self.max_velocity = 1.0

    def initialize(self):
        self.population = [
            [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
            for _ in range(self.population_size)
        ]
        self.velocities = [
            [random.uniform(self.min_velocity, self.max_velocity) for _ in range(self.dim)]
            for _ in range(self.population_size)
        ]

        self.pbest_positions = self.population[:]
        self.pbest_values = [self.evaluate(individual) for individual in self.population]

        self.gbest_value = min(self.pbest_values)
        best_index = self.pbest_values.index(self.gbest_value)
        self.gbest_position = self.pbest_positions[best_index]

    def evaluate(self, individual):
        """Evaluate an individual's fitness."""
        if self.evals >= self.max_evals:
            return float('inf')
        self.evals += 1
        return self.func(individual)

    def optimize(self):
        self.initialize()

        while self.evals < self.max_evals:
            for i in range(self.population_size):
                for d in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()

                    # Update velocity
                    self.velocities[i][d] = (
                        self.w * self.velocities[i][d]
                        + self.c1 * r1 * (self.pbest_positions[i][d] - self.population[i][d])
                        + self.c2 * r2 * (self.gbest_position[d] - self.population[i][d])
                    )
                    # Constrain velocity
                    self.velocities[i][d] = max(
                        min(self.velocities[i][d], self.max_velocity), self.min_velocity
                    )

                    # Update position
                    self.population[i][d] += self.velocities[i][d]
                    # Enforce bounds
                    self.population[i][d] = max(
                        min(self.population[i][d], self.bounds[d][1]), self.bounds[d][0]
                    )

                # Evaluate individual
                fitness = self.evaluate(self.population[i])

                # Update personal best
                if fitness < self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest_positions[i] = self.population[i][:]

                # Update global best
                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = self.population[i][:]

            if self.evals >= self.max_evals:
                break

        return self.gbest_value, self.gbest_position

def run():
    dimensions = 5
    bounds = [(-10, 10) for _ in range(dimensions)]
    maximum_evaluations = 1000

    # Run PSO with the given constrained function
    optimizer = ParticleSwarmOptimizer(constrained_function, dimensions, bounds, maximum_evaluations)
    best_value, best_position = optimizer.optimize()
    print(f"Best Value: {best_value}\nBest Position: {best_position}")

if __name__ == "__main__":
    run()
