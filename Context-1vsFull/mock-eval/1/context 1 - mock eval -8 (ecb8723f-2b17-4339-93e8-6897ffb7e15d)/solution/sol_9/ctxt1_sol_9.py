import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=dim
        )
        self.velocity = np.random.uniform(-1, 1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

class PSO:
    evals = 0  # Class variable to track function evaluation count

    def __init__(self, func, dim, bounds, max_evals, num_particles=30, w=0.5, c1=1.5, c2=1.5):
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
                fitness = self.evaluate(particle.position)

                # Update personal best
                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = np.copy(particle.position)

                # Update global best if some improvement is found
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

                if PSO.evals >= self.max_evals:
                    break

                self.update_velocity_and_position(particle)

        return self.global_best_value, self.global_best_position

    def update_velocity_and_position(self, particle):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)

        cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
        social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
        particle.position += particle.velocity

        # Apply bounds to keep positions within the defined space
        for i in range(self.dim):
            if particle.position[i] < self.bounds[i][0] or particle.position[i] > self.bounds[i][1]:
                particle.velocity[i] *= -1  # Reflect the velocity if out of bounds
            particle.position[i] = np.clip(particle.position[i], self.bounds[i][0], self.bounds[i][1])

    def evaluate(self, position):
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
#
#In this updated version, I made the following improvements:
#
#- Initialization of particle velocities to small random values instead of zeros, to encourage better exploration from the start.
#- Added a reflection mechanism for velocities when particles move out of bounds, which helps in exploring edge areas of the search space more effectively.
#- Ensured positions are clipped to stay within the defined bounds.
#- Updated comments for clarity and maintainability of the code. 
#
#The code follows clean practices and leverages numpy efficiently, ensuring performance and readability.
