import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')

    # [Algorithm body: Refined PSO incorporating Dynamic Swarm Strategy and Enhanced Reinitialization]

    num_particles = 30
    initial_w = 0.9
    final_w = 0.4
    c1 = 1.5
    c2 = 1.5
    velocity_clamp_factor = 0.1
    reinit_fraction = 0.2  # Reinitialize 20% of particles

    def adaptive_inertia_weight(evals, max_evals):
        """Nonlinear inertia reduction for adaptive exploitation/exploration."""
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

    # Begin iterations
    while evals < max_evals:
        w = adaptive_inertia_weight(evals, max_evals)

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

        if evals % (max_evals // 5) == 0:
            # Reinitialize a portion of particles to avoid early stagnation
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
#1. **Adaptive Inertia Weight:** Maintains use of a nonlinear function for inertia weight, focusing on exploratory and exploitative balance dynamically.
#2. **Strategic Reinitialization:** Regularly refreshes a portion of particles to maintain swarm diversity and escape premature convergence.
#3. **Enhanced Velocity Handling:** Ensures velocities are kept within a range that prevents particles from being lost outside the feasible search space.
#4. **Parameter Optimization:** Tweaked the parameters to achieve a better compromise between exploration and convergence speed, incorporating feedback for more robust performance across varied problem spaces.
