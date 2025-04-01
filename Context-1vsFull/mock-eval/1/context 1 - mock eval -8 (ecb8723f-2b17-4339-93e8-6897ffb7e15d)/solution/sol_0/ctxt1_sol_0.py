class Particle:
    def __init__(self, dim, bounds):
        self.position = [np.random.uniform(low, high) for low, high in bounds]
        self.velocity = [0.0] * dim
        self.best_position = list(self.position)
        self.best_value = float('inf')

def run(func, dim, bounds, max_evals):
    import numpy as np

    num_particles = 30
    w = 0.5  # inertia weight
    c1 = 1.5  # cognitive coefficient
    c2 = 1.5  # social coefficient
    
    particles = [Particle(dim, bounds) for _ in range(num_particles)]
    
    global_best_value = float('inf')
    global_best_position = None
    evals = 0
    
    while evals < max_evals:
        for particle in particles:
            # Evaluate the fitness
            fitness = func(particle.position)
            evals += 1

            # Update personal best
            if fitness < particle.best_value:
                particle.best_value = fitness
                particle.best_position = list(particle.position)
            
            # Update global best
            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = list(particle.position)
        
        if evals >= max_evals:
            break

        # Update velocity and position
        for particle in particles:
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)

            cognitive_velocity = c1 * r1 * (np.array(particle.best_position) - np.array(particle.position))
            social_velocity = c2 * r2 * (np.array(global_best_position) - np.array(particle.position))

            particle.velocity = w * np.array(particle.velocity) + cognitive_velocity + social_velocity
            particle.position += particle.velocity

            # Apply boundary constraints
            for i in range(dim):
                particle.position[i] = np.clip(particle.position[i], bounds[i][0], bounds[i][1])

    return global_best_value
