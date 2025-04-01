import random

def run(func, dim, bounds, max_evals):
    class Particle:
        def __init__(self):
            self.position = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            self.velocity = [random.uniform(-1, 1) for _ in range(dim)]
            self.best_position = self.position[:]
            self.best_fitness = None
        
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
            if particle.position[i] < bounds[i][0]:
                particle.position[i] = bounds[i][0]
            if particle.position[i] > bounds[i][1]:
                particle.position[i] = bounds[i][1]

    def evaluate(position):
        nonlocal evals
        evals += 1
        return func(position)

    evals = 0
    num_particles = 30
    inertia_weight = 0.7
    cognitive_coef = 1.5
    social_coef = 2.0

    particles = initialize_particles(num_particles)
    best = float('inf')
    global_best_position = particles[0].position[:]

    for particle in particles:
        if particle.best_fitness < best:
            best = particle.best_fitness
            global_best_position = particle.best_position[:]

    while evals < max_evals:
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
