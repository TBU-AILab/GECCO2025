import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Enhanced PSO with Nonlinear Inertia Reduction and Reinitialization Strategy]

    num_particles = 30
    initial_w = 0.9
    final_w = 0.2  # Lower final inertia weight for tighter convergence
    c1 = 1.5      # Moderated cognitive component
    c2 = 1.5      # Moderated social component
    velocity_clamp_factor = 0.2  
    reinit_fraction = 0.2  # Fraction of particles to reinitialize periodically

    def nonlinear_inertia_weight(evals, max_evals):
        """Nonlinear decay of inertia to focus gradually more on exploitation."""
        return final_w + (initial_w - final_w) * ((max_evals - evals) / max_evals) ** 2

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

    while evals < max_evals:
        w = nonlinear_inertia_weight(evals, max_evals)

        for particle in particles:
            current_fitness = func(particle['position'])
            evals += 1

            if current_fitness < particle['best_fitness']:
                particle['best_fitness'] = current_fitness
                particle['best_position'] = particle['position'][:]

            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle['position'][:]

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = particle['position'][:]

            if evals >= max_evals:
                return best_fitness

        for particle in particles:
            for i in range(dim):
                r1 = random.random()
                r2 = random.random()

                new_velocity = (w * particle['velocity'][i] +
                                c1 * r1 * (particle['best_position'][i] - particle['position'][i]) +
                                c2 * r2 * (global_best_position[i] - particle['position'][i]))

                max_velocity = (bounds[i][1] - bounds[i][0]) * velocity_clamp_factor
                particle['velocity'][i] = max(-max_velocity, min(new_velocity, max_velocity))

                particle['position'][i] += particle['velocity'][i]

                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]

        if evals % (int(max_evals * reinit_fraction)) == 0:
            num_reinit = int(num_particles * reinit_fraction)
            reinit_indices = random.sample(range(num_particles), num_reinit)
            for idx in reinit_indices:
                particles[idx]['position'] = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
                particles[idx]['velocity'] = [random.uniform(-(bounds[i][1] - bounds[i][0]) * velocity_clamp_factor,
                                                             (bounds[i][1] - bounds[i][0]) * velocity_clamp_factor) for i in range(dim)]
                particles[idx]['best_position'] = particles[idx]['position']
                particles[idx]['best_fitness'] = float('inf')

    return best_fitness
#
#### Key Enhancements:
#1. **Nonlinear Inertia Weight**: Utilizes a nonlinear reduction in inertia weight to maintain a strong exploration phase initially, then focusing on exploitation as the evaluations progress.
#2. **Moderate Cognitive and Social Weights**: Tweaked these values to better balance individual discovery and group convergence.
#3. **Periodic Reinitialization Mechanism**: Regularly resets a subset of particles to combat premature convergence and to maintain diversity within the swarm.
