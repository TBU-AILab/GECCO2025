import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Refined Hybrid PSO with Neighborhood Search and Adaptive Mechanisms]

    num_particles = 40  # Increased number of particles for comprehensive exploration
    initial_w = 0.9     # Initial inertia weight
    final_w = 0.4       # Final inertia weight for convergence acceleration
    c1 = 2.0            # Cognitive (particle memory) component
    c2 = 2.0            # Social (swarm memory) component
    velocity_clamp_factor = 0.1  # Limit for velocity relative to the problem's bounding box

    def update_inertia_weight(evals, max_evals):
        """ Linearly interpolate inertia weight from initial to final value. """
        return initial_w - (initial_w - final_w) * float(evals) / max_evals

    # Initialize particles
    particles = [{'position': [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)],
                  'velocity': [random.uniform(-(bounds[i][1] - bounds[i][0]) * velocity_clamp_factor,
                                              (bounds[i][1] - bounds[i][0]) * velocity_clamp_factor) for i in range(dim)],
                  'best_position': None,
                  'best_fitness': float('inf')}
                 for _ in range(num_particles)]

    # Initialize global best
    global_best_position = None
    global_best_fitness = float('inf')

    # Begin iterations
    while evals < max_evals:
        w = update_inertia_weight(evals, max_evals)
        
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

                # Update velocity with dynamic inertia, cognitive, and social components
                new_velocity = (w * particle['velocity'][i] +
                                c1 * r1 * (particle['best_position'][i] - particle['position'][i]) +
                                c2 * r2 * (global_best_position[i] - particle['position'][i]))
                
                # Velocity clamping
                max_velocity = (bounds[i][1] - bounds[i][0]) * velocity_clamp_factor
                particle['velocity'][i] = max(-max_velocity, min(new_velocity, max_velocity))

                # Update position with clamped velocity and adaptive temperature influence
                annealing_noise = (random.uniform(-0.5, 0.5) * temperature)
                particle['position'][i] += particle['velocity'][i] + annealing_noise

                # Clamping position within bounds
                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]
        
        # Optional: Neighborhood-based refinement
        if evals < max_evals:  # Ensure this doesn't exceed eval count
            for particle in particles:
                new_position = [(p + random.uniform(-0.05, 0.05) * (b[1] - b[0])) for p, b in zip(particle['position'], bounds)]
                
                # Evaluate fitness for the new position
                new_fitness = func(new_position)
                evals += 1

                # Decide if the new position becomes the current position
                if new_fitness < particle['best_fitness']:
                    particle['position'] = new_position
                    particle['best_fitness'] = new_fitness
                    particle['best_position'] = new_position
                    if new_fitness < global_best_fitness:
                        global_best_fitness = new_fitness
                        global_best_position = new_position
                
                # Early stop if max_evals is reached during neighborhood search
                if evals >= max_evals:
                    break

    # Return the fitness of the best found solution
    return best_fitness
#
#In this refined approach, I've made the following enhancements:
#1. Increased the particle swarm size further to maximize search space coverage.
#2. Implemented a linearly decaying inertia weight to shift emphasis from exploration to exploitation over time.
#3. Introduced velocity clamping to maintain control over particle movement, assisting in fine-tuning exploration.
#4. Integrated an optional neighborhood search mechanism that leverages local exploration for potential improvements, applied briefly until evals exceed max evaluations.
#5. Adaptive temperature factor continues to introduce small-scale perturbation, aiding in escaping local optima.
