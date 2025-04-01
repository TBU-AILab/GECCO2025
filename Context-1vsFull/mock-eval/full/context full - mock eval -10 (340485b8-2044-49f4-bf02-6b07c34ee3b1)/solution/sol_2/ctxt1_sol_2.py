import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Enhanced PSO with Dynamic Parameters and Adaptive Annealing]

    num_particles = 20  # Further increased number of particles for diversity
    initial_w = 0.9  # Initial inertia weight
    final_w = 0.4    # Final inertia weight for better convergence
    c1 = 2.0         # Cognitive weight
    c2 = 2.0         # Social weight

    def update_inertia_weight(evals, max_evals, initial_w, final_w):
        """ Linearly decreases inertia weight from initial to final value. """
        return initial_w - (initial_w - final_w) * (evals / max_evals)

    # Initialize particles
    particles = [{'position': [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)],
                  'velocity': [random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) for i in range(dim)],
                  'best_position': None,
                  'best_fitness': float('inf')}
                 for _ in range(num_particles)]

    # Initialize global best
    global_best_position = None
    global_best_fitness = float('inf')

    # Begin iterations
    while evals < max_evals:
        w = update_inertia_weight(evals, max_evals, initial_w, final_w)
        
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
        
        temperature = 1.0 - evals / max_evals
        for particle in particles:
            for i in range(dim):
                r1 = random.random()
                r2 = random.random()

                # Dynamic inertia in velocity
                particle['velocity'][i] = (w * particle['velocity'][i] +
                                           c1 * r1 * (particle['best_position'][i] - particle['position'][i]) +
                                           c2 * r2 * (global_best_position[i] - particle['position'][i]))

                # Adaptive temperature influence
                annealing_noise = (random.uniform(-0.5, 0.5) * temperature)
                particle['position'][i] += particle['velocity'][i] + annealing_noise

                # Clamping position within bounds
                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]

    # Return the fitness of the best found solution
    return best_fitness
#
#In this enhanced implementation, I've: 
#1. Increased the number of particles even further to enhance the search space coverage.
#2. Introduced dynamic inertia weight that decreases linearly over time, balancing exploration and exploitation.
#3. Ensured the initial velocity of particles is proportionate to the search space width to improve initial exploration.
#4. Applied adaptive annealing influences to fine-tune exploration, introducing small, diminishing random changes to avoid local traps.
#5. Maintained boundary position limits to ensure solutions remain feasible throughout the optimization process.
