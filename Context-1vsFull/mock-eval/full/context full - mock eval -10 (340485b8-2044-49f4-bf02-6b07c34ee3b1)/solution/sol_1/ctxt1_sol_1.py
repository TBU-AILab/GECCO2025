import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Improved Hybrid PSO with Adaptive Annealing and Boundary Handling]
    
    num_particles = 15  # Increased number of particles for broader exploration
    w = 0.7  # Inertia weight
    c1 = 1.5  # Cognitive (particle) weight
    c2 = 1.5  # Social (global) weight

    # Initialize particles
    particles = [{'position': [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)],
                  'velocity': [0.0 for _ in range(dim)],
                  'best_position': None,
                  'best_fitness': float('inf')}
                 for _ in range(num_particles)]

    # Initialize global best
    global_best_position = None
    global_best_fitness = float('inf')

    # Begin iterations
    while evals < max_evals:
        for particle in particles:
            # Evaluate fitness
            current_fitness = func(particle['position'])
            evals += 1

            # Update personal best
            if current_fitness < particle['best_fitness']:
                particle['best_fitness'] = current_fitness
                particle['best_position'] = particle['position'][:]

            # Update global best
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle['position'][:]

            # Update best solution found
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = particle['position'][:]

            # Early stop if max_evals is reached
            if evals >= max_evals:
                break
        
        # Adaptive annealing-influenced velocity and position update
        temperature = 1.0 - evals / max_evals
        for particle in particles:
            for i in range(dim):
                r1 = random.random()
                r2 = random.random()

                # Update velocity with inertia, cognitive, and social components
                particle['velocity'][i] = (w * particle['velocity'][i] +
                                           c1 * r1 * (particle['best_position'][i] - particle['position'][i]) +
                                           c2 * r2 * (global_best_position[i] - particle['position'][i]))

                # Simulated annealing with temperature-dependent noise
                annealing_noise = (random.uniform(-0.5, 0.5) * temperature)
                particle['position'][i] += particle['velocity'][i] + annealing_noise
                
                # Clamping position within bounds
                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                    particle['velocity'][i] = 0  # Reflect velocity to prevent leaving bounds
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]
                    particle['velocity'][i] = 0  # Reflect velocity to prevent leaving bounds

    # Return the fitness of the best found solution
    return best_fitness
#
#In this improved version, I have increased the number of particles for enhanced exploration of the search space. The inertia, cognitive, and social weights are combined with a temperature-dependent annealing noise that adapts over time, promoting both exploration and exploitation. Boundary handling ensures particles do not leave the feasible region by reflecting the velocity to zero when boundaries are hit, thus improving stability and convergence.
