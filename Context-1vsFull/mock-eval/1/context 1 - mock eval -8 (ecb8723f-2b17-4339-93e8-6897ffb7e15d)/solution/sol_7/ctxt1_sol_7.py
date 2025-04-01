import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=dim
        )
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

class PSO:
    evals = 0  # Class variable to track the number of function evaluations globally
    
    def __init__(self, func, dim, bounds, max_evals, num_particles=30, w=0.7, c1=1.4, c2=1.4):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_value = float('inf')
        self.global_best_position = np.zeros(dim)

    def optimize(self):
        while PSO.evals < self.max_evals:
            for particle in self.particles:
                self.update_particle(particle)

                if PSO.evals >= self.max_evals:
                    break

                self.update_velocity_and_position(particle)

        return self.global_best_value, self.global_best_position

    def update_particle(self, particle):
        fitness = self.evaluate(particle.position)

        if fitness < particle.best_value:
            particle.best_value = fitness
            particle.best_position = np.copy(particle.position)

        if fitness < self.global_best_value:
            self.global_best_value = fitness
            self.global_best_position = np.copy(particle.position)

    def update_velocity_and_position(self, particle):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)

        cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
        social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)

        particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
        particle.position += particle.velocity

        # Apply bounds to ensure particles stay within the defined space
        particle.position = np.clip(
            particle.position, [b[0] for b in self.bounds], [b[1] for b in self.bounds]
        )

    def evaluate(self, position):
        """ Evaluate the function and increase evaluation counter """
        result = self.func(position)
        PSO.evals += 1
        return result

def inverted_sphere(position):
    return -sum(x**2 for x in position)

def run(func, dim, bounds, max_evals):
    PSO.evals = 0  # Reset evals for each run
    optimizer = PSO(func=func, dim=dim, bounds=bounds, max_evals=max_evals)
    best_value, best_position = optimizer.optimize()

    return best_value, best_position

if __name__ == '__main__':
    dim = 5
    bounds = [(-5.0, 5.0)] * dim
    max_evals = 1000

    best_value, best_position = run(inverted_sphere, dim, bounds, max_evals)
    print("Best Value:", best_value)
    print("Best Position:", best_position)
#This improved solution introduces slight adjustments to the PSO parameters, aiming for better convergence. Additionally, I ensured the structure remains clear, logically consistent, and aligned with the Python best practices.
