import random
import math

def run(func, dim, bounds, max_evals):
    class Particle:
        def __init__(self):
            self.position = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            self.velocity = [random.uniform(-1, 1) for _ in range(dim)]
            self.best_position = self.position[:]
            self.best_fitness = float('inf')

    def initialize_particles(num_particles):
        particles = [Particle() for _ in range(num_particles)]
        for particle in particles:
            fitness = evaluate(particle.position)
            particle.best_fitness = fitness
            particle.best_position = particle.position[:]
        return particles

    def update_velocity(particle, global_best_position, inertia_weight, cognitive_coef, social_coef):
        for i in range(dim):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = cognitive_coef * r1 * (particle.best_position[i] - particle.position[i])
            social_velocity = social_coef * r2 * (global_best_position[i] - particle.position[i])
            particle.velocity[i] = (inertia_weight * particle.velocity[i] +
                                    cognitive_velocity +
                                    social_velocity)

    def update_position(particle):
        for i in range(dim):
            particle.position[i] += particle.velocity[i]
            # Enforce bounds
            particle.position[i] = min(max(particle.position[i], bounds[i][0]), bounds[i][1])

    def evaluate(position):
        nonlocal evals
        evals += 1
        return func(position)

    def adapt_parameters(iteration, max_iterations):
        # Implement dynamic parameter adjustment based on iteration
        inertia_weight = 0.9 - 0.5 * (iteration / max_iterations)
        cognitive_coef = 1.5 + 0.5 * math.sin(2 * math.pi * iteration / max_iterations)
        social_coef = 1.5 + 0.5 * math.cos(2 * math.pi * iteration / max_iterations)
        return inertia_weight, cognitive_coef, social_coef

    evals = 0
    num_particles = 60  # Further increased number of particles
    particles = initialize_particles(num_particles)
    best = float('inf')
    global_best_position = particles[0].position[:]

    for particle in particles:
        if particle.best_fitness < best:
            best = particle.best_fitness
            global_best_position = particle.best_position[:]

    max_iterations = max_evals // num_particles

    for iteration in range(max_iterations):
        inertia_weight, cognitive_coef, social_coef = adapt_parameters(iteration, max_iterations)
        
        for particle in particles:
            update_velocity(particle, global_best_position, inertia_weight, cognitive_coef, social_coef)
            update_position(particle)
            fitness = evaluate(particle.position)
            
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position[:]
            
            if fitness < best:
                best = fitness
                global_best_position = particle.position[:]

    return best
#
#### Improvements Made:
#1. **Increased Number of Particles**: Further increased to 60 to improve the exploration capability of the algorithm.
#
#2. **Dynamic Parameter Adjustment**: Introduced a dynamic adaptation scheme for `inertia_weight`, `cognitive_coef`, and `social_coef` to ensure better balance between exploration and exploitation across iterations.
#
#3. **Improved Boundary Handling**: Streamlined boundary checks within the `update_position` function for clarity and efficiency.
#
#These changes aim to improve the mean results across functions, attempting to address weaknesses highlighted by the Wilcoxon rank sum test feedback by improving convergence rates and exploring diverse solution spaces.
