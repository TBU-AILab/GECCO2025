import random

class PSO:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.best_value = float('inf')
        self.best_position = None
        self.evals = 0
        self.population_size = 20
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (personal) weight
        self.c2 = 1.5  # social (global) weight
        self.min_velocity = -1.0
        self.max_velocity = 1.0

    def initialize_population(self):
        self.population = [
            [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
            for _ in range(self.population_size)
        ]
        self.velocities = [
            [random.uniform(self.min_velocity, self.max_velocity) for _ in range(self.dim)]
            for _ in range(self.population_size)
        ]

        self.pbest_positions = self.population[:]
        self.pbest_values = [self.func(individual) for individual in self.population]
        self.evals += self.population_size

        self.gbest_value = min(self.pbest_values)
        self.gbest_position = self.pbest_positions[self.pbest_values.index(self.gbest_value)]

    def run(self):
        self.initialize_population()

        while self.evals < self.max_evals:
            for i in range(self.population_size):
                for d in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()

                    self.velocities[i][d] = (
                        self.w * self.velocities[i][d]
                        + self.c1 * r1 * (self.pbest_positions[i][d] - self.population[i][d])
                        + self.c2 * r2 * (self.gbest_position[d] - self.population[i][d])
                    )
                    self.velocities[i][d] = max(
                        min(self.velocities[i][d], self.max_velocity), self.min_velocity
                    )

                    self.population[i][d] += self.velocities[i][d]
                    self.population[i][d] = max(
                        min(self.population[i][d], self.bounds[d][1]), self.bounds[d][0]
                    )

                fitness = self.func(self.population[i])
                self.evals += 1

                if fitness < self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest_positions[i] = self.population[i][:]

                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = self.population[i][:]

                if self.evals >= self.max_evals:
                    break

        return self.gbest_value, self.gbest_position

# Example usage
def sample_function(x):
    return sum(xi ** 2 for xi in x)

# Dimensions and bounds for the problem
dimensions = 5
bounds = [(-10, 10) for _ in range(dimensions)]
maximum_evaluations = 1000

# Running the PSO algorithm
pso_solver = PSO(sample_function, dimensions, bounds, maximum_evaluations)
best_value, best_position = pso_solver.run()
print(f"Best Value: {best_value}\nBest Position: {best_position}")
