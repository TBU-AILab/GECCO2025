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
            personal_influence = cognitive_coef * r1 * (particle.best_position[i] - particle.position[i])
            social_influence = social_coef * r2 * (global_best_position[i] - particle.position[i])
            particle.velocity[i] = (inertia_weight * particle.velocity[i] +
                                    personal_influence +
                                    social_influence)

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
        inertia_weight = 0.7 - 0.5 * (iteration / max_iterations)  # More aggressive reduction
        cognitive_coef = 1.5  # Heighten the personal search influence
        social_coef = 2.5  # Heightened social effect to leverage group insights
        return inertia_weight, cognitive_coef, social_coef

    evals = 0
    num_particles = 80  # Increased particle count for broader solution exploration
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
                global_best_position = particle.best_position[:]

    return best
#
#### Key Adjustments:
#
#1. **Inertia Weight Adaptation**: The inertia weight's range is more aggressively reduced `(0.7 to 0.2)`, favoring a quicker transition from exploration to exploitation as the process progresses.
#
#2. **Coefficients Tuning**: The `cognitive_coef` is set to `1.5`, and `social_coef` is increased to `2.5` to emphasize both personal experience and swarm collaboration, enhancing balance and improvement opportunities.
#
#3. **Particle Count**: Increased number of particles to `80` to widen the diversity of sampled solutions, increasing the likelihood of finding optimal or near-optimal solutions, particularly addressing shortcomings noted in specific functions.
#
#These improvements aim to harness swarm dynamics more effectively to enhance solver performance across the trialed functions, aiming for reduced disparities with best-so-far solutions.
