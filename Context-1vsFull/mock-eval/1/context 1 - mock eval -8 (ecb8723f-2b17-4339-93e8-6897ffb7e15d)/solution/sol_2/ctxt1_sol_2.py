import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

class PSO:
    def __init__(self, func, dim, bounds, max_evals, num_particles=30, w=0.5, c1=1.5, c2=1.5):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.evals = 0

        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best_value = float('inf')
        self.global_best_position = np.zeros(dim)

    def optimize(self):
        while self.evals < self.max_evals:
            for particle in self.particles:
                fitness = self.func(particle.position)
                self.evals += 1

                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            if self.evals >= self.max_evals:
                break

            for particle in self.particles:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)

                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                particle.position += particle.velocity

                for i in range(self.dim):
                    if particle.position[i] < self.bounds[i][0]:
                        particle.position[i] = self.bounds[i][0]
                    elif particle.position[i] > self.bounds[i][1]:
                        particle.position[i] = self.bounds[i][1]

        return self.global_best_value, self.global_best_position

def inverted_sphere(position):
    return -sum(x**2 for x in position)

def run():
    dim = 5
    bounds = [(-5.0, 5.0)] * dim
    max_evals = 1000

    optimizer = PSO(func=inverted_sphere, dim=dim, bounds=bounds, max_evals=max_evals)
    best_value, best_position = optimizer.optimize()

    return best_value, best_position

if __name__ == '__main__':
    best_value, best_position = run()
    print("Best Value:", best_value)
    print("Best Position:", best_position)
