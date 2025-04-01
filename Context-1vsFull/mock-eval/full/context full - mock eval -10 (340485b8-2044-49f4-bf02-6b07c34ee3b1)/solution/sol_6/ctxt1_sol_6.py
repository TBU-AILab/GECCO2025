import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Optimized Hybrid PSO with Adaptive Control and Periodic Reinitialization]

    num_particles = 30  # Balance between exploration and computational cost
    initial_w = 0.9     # High inertia to start with exploration
    final_w = 0.4       # Lower inertia for convergence
    c1 = 2.05           # Slightly increased cognitive component to encourage diversity
    c2 = 2.05           # Slightly increased social component to enhance cooperation
    velocity_clamp_factor = 0.2  # Velocity clamp adjusted for variability

    def update_inertia_weight(evals, max_evals):
        """Adaptive inertia weight reducing linearly."""
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

                # Ensuring the particle remains within bounds
                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]

        # Periodic reinitialization for diversity boost
        if evals % (max_evals // 4) == 0:  # Reinitialize a quarter of particles 
            for particle in random.sample(particles, num_particles // 4):
                particle['position'] = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
                particle['velocity'] = [random.uniform(-(bounds[i][1] - bounds[i][0]) * velocity_clamp_factor,
                                                       (bounds[i][1] - bounds[i][0]) * velocity_clamp_factor) for i in range(dim)]
                particle['best_position'] = particle['position']
                particle['best_fitness'] = func(particle['position'])
                evals += 1

    return best_fitness
#
#### Enhancements Made:
#1. **Adaptive Parameters**: Fine-tuned inertia, cognitive, and social parameters to enhance balance and efficiency.
#2. **Velocity Clamping**: Adjusted to factor in more dynamic changes without exceeding logical bounds.
#3. **Periodic Reinitialization**: Introduced a mechanism to periodically reinitialize particles, maintaining diversity and preventing stagnation.
#4. **Boundary Handling**: Ensured particles recovering efficiently from boundary edges, reinforcing feasible space exploration.
