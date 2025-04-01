import numpy as np

class PSOOptimizer:
    def __init__(self, func, dim, bounds, population_size=20, inertia_weight=0.7, 
                 cognitive_coefficient=1.5, social_coefficient=1.5, max_evals=1000, seed=42):
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.max_evals = max_evals
        self.evals = 0
        np.random.seed(seed)
        
        # Initialize population and velocities
        self.population, self.velocities = self.initialize_population()
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = self.evaluate_population(self.population)
        
        # Identify global best position and fitness
        self.global_best_position, self.global_best_fitness = self.find_global_best(self.personal_best_positions, self.personal_best_fitness)

    def initialize_population(self):
        """ Initialize population and velocities within the given bounds. """
        lower_bounds, upper_bounds = self.bounds.T
        population = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
        velocities = np.zeros_like(population)
        return population, velocities

    def evaluate_population(self, population):
        """ Evaluate the function for each individual in the population. """
        return np.array([self.func(individual) for individual in population])
    
    def update_personal_best(self, population, fitness):
        """ Update personal best positions and fitness for each individual. """
        update_mask = fitness < self.personal_best_fitness
        self.personal_best_positions[update_mask] = population[update_mask]
        self.personal_best_fitness[update_mask] = fitness[update_mask]

    def find_global_best(self, personal_best_positions, personal_best_fitness):
        """ Determine the global best position and fitness in the population. """
        best_idx = np.argmin(personal_best_fitness)
        return personal_best_positions[best_idx], personal_best_fitness[best_idx]

    def run(self):
        """ Main optimization loop of the PSO algorithm. """
        while self.evals < self.max_evals:
            for i in range(self.population_size):
                if self.evals >= self.max_evals:
                    break

                # Generate random coefficients for velocity update
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                # Update velocity
                self.velocities[i] = (self.inertia_weight * self.velocities[i] + 
                                      self.cognitive_coefficient * r1 * (self.personal_best_positions[i] - self.population[i]) +
                                      self.social_coefficient * r2 * (self.global_best_position - self.population[i]))

                # Update position with clipping to maintain bounds
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], *self.bounds.T)

                # Evaluate fitness
                fitness = self.func(self.population[i])
                self.evals += 1

                # Update personal bests
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position, self.global_best_fitness, self.evals

# Example usage
def example_func(x):
    """ Example function: Sphere function """
    return np.sum(x ** 2)

if __name__ == '__main__':
    dim = 5
    bounds = [(-5, 5)] * dim
    max_evals = 1000
    optimizer = PSOOptimizer(example_func, dim, bounds, max_evals=max_evals)
    best_position, best_fitness, total_evals = optimizer.run()
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")
    print(f"Total Evaluations: {total_evals}")
