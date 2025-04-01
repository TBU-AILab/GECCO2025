import random

def evaluate_decorator(func):
    ''' Decorator to count the evaluations of the cost function '''
    def wrapper(*args, **kwargs):
        wrapper.evals += 1
        return func(*args, **kwargs)
    wrapper.evals = 0
    wrapper.__name__ = func.__name__
    return wrapper

@evaluate_decorator
def inverted_sphere(x):
    ''' Inverted Sphere function to maximize '''
    return -sum(xi ** 2 for xi in x)

class ParticleSwarmOptimizer:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

        self.population_size = 20
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.min_velocity = -1.0
        self.max_velocity = 1.0

    def initialize(self):
        ''' Initialize the population and velocities '''
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

        self.gbest_value = max(self.pbest_values)
        best_index = self.pbest_values.index(self.gbest_value)
        self.gbest_position = self.pbest_positions[best_index]

    def optimize(self):
        ''' The main optimization loop '''
        self.initialize()

        while self.func.evals < self.max_evals:
            for i in range(self.population_size):
                for d in range(self.dim):
                    r1, r2 = random.random(), random.random()

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

                if fitness > self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest_positions[i] = self.population[i][:]

                if fitness > self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = self.population[i][:]

            if self.func.evals >= self.max_evals:
                break

        return -self.gbest_value, self.gbest_position

def run(func=None, dim=None, bounds=None, max_evals=None):
    if func is None:
        func = inverted_sphere
    if dim is None:
        dim = 5
    if bounds is None:
        bounds = [(-10, 10) for _ in range(dim)]
    if max_evals is None:
        max_evals = 1000

    optimizer = ParticleSwarmOptimizer(func, dim, bounds, max_evals)
    best_value, best_position = optimizer.optimize()
    print(f"Best Value: {best_value}\nBest Position: {best_position}\nFunction Evaluations: {func.evals}")

if __name__ == "__main__":
    run()
