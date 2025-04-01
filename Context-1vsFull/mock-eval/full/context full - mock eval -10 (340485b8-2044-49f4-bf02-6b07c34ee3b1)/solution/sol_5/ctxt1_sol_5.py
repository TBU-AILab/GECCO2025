import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Advanced Hybrid PSO with Adaptive Mechanisms and Local Refinement]

    num_particles = 30  # Optimal number for balance between exploration and execution speed
    initial_w = 0.9     # Initial inertia weight for exploration
    final_w = 0.4       # Final inertia weight for exploitation
    c1 = 2.0            # Cognitive component weight for personal influence
    c2 = 2.0            # Social component weight for global influence
    velocity_clamp_factor = 0.1  # Clamp velocity as fraction of bounds' width

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
            
            if evals >= max_evals:
                return best_fitness

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

                # Update position with clamped velocity
                particle['position'][i] += particle['velocity'][i]

                # Ensuring positions remain within bounds
                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]

        # Local refinement using Gaussian perturbation
        for particle in particles:
            candidate_position = [p + random.gauss(0, temperature * (b[1] - b[0])) for p, b in zip(particle['position'], bounds)]
            candidate_position = [max(min(p, b[1]), b[0]) for p, b in zip(candidate_position, bounds)]
            
            # Evaluate new candidate
            candidate_fitness = func(candidate_position)
            evals += 1

            # Update if better
            if candidate_fitness < particle['best_fitness']:
                particle['best_fitness'] = candidate_fitness
                particle['best_position'] = candidate_position
                if candidate_fitness < global_best_fitness:
                    global_best_fitness = candidate_fitness
                    global_best_position = candidate_position

    return best_fitness
#
#### Enhancements Made:
#1. **Dynamic Inertia**: Maintained linearly decreasing inertia to balance exploration and exploitation.
#2. **Velocity Clamping**: Ensures that particles do not deviate too much, preserving the search within a realistic search space.
#3. **Local Refinement with Gaussian Perturbation**: Introduces a new local search strategy using Gaussian noise, which adaptively decreases the search space over time.
#4. **Boundary Control**: Adjusts particles back into bounds immediately after updating positions, ensuring no unintended boundary violations.
#
