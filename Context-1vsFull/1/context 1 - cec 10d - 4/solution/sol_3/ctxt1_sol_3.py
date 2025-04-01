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
        # Implement a more dynamic parameter adjustment
        inertia_weight = 0.9 - 0.4 * (iteration / max_iterations)
        cognitive_coef = 2.0  # Fixed cognitive coefficient for stronger local search
        social_coef = 2.0  # Fixed social coefficient for stronger global search
        return inertia_weight, cognitive_coef, social_coef

    evals = 0
    num_particles = 60  # Further increased number of particles helps improve search space exploration
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
#### Key Improvements:
#1. **Parameter Tuning**: I refined the parameter tuning by setting `cognitive_coef` and `social_coef` to 2.0 to reinforce the balance between individual exploration and team influence, aiming to stabilise and effectively utilize swarm intelligence.
#
#2. **Dynamic Inertia Weight**: Fine-tuned the adaptation of inertia weight by decreasing its range for better fine-tuning, thus encouraging a steady transition from global to local search as iterations progress.
#
#These enhancements aim to strengthen the optimizer's performance across functions where improvements were necessary, whilst consolidating gains from previous improvements on multiple functions as per the feedback stated.
