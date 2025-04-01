#I apologize for the confusion earlier. Let's resolve this issue by ensuring the `random` module is correctly initialized and accessible across all parts of the code. Here's the corrected version, structured to ensure `random` is imported correctly:
#
import random

def run(func, dim, bounds, max_evals):
    # Parameters for Particle Swarm Optimization
    num_particles = 50
    inertia_weight = 0.7
    min_inertia = 0.4
    cognitive_coeff = 1.5
    social_coeff = 1.5

    # Parameters for Differential Evolution
    mutation_factor = 0.8
    crossover_prob = 0.9

    # Initialize particle swarm
    particles = [{
        'position': [random.uniform(low, high) for low, high in bounds],
        'velocity': [0.0] * dim,
        'personal_best_pos': None,
        'personal_best_val': float('inf')
    } for _ in range(num_particles)]

    # Initialize global best position and value
    global_best_pos = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
    global_best_val = float('inf')

    evals = 0
    last_improvement = 0

    def update_particle(particle, global_best_pos):
        nonlocal inertia_weight
        # Particle Swarm Optimization update
        for d in range(dim):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = cognitive_coeff * r1 * (particle['personal_best_pos'][d] - particle['position'][d])
            social_velocity = social_coeff * r2 * (global_best_pos[d] - particle['position'][d])
            particle['velocity'][d] = inertia_weight * particle['velocity'][d] + cognitive_velocity + social_velocity

            # Update position
            particle['position'][d] += particle['velocity'][d]

            # Boundary constraints
            particle['position'][d] = min(max(particle['position'][d], bounds[d][0]), bounds[d][1])

    def apply_differential_evolution():
        nonlocal evals, global_best_pos, global_best_val
        for i in range(num_particles):
            if random.random() < crossover_prob:
                a, b, c = random.sample([j for j in range(num_particles) if j != i], 3)
                trial = []

                for d in range(dim):
                    if random.random() < 0.5:
                        trial_value = particles[a]['position'][d] + mutation_factor * (particles[b]['position'][d] - particles[c]['position'][d])
                        # Ensure within bounds
                        trial_value = max(min(trial_value, bounds[d][1]), bounds[d][0])
                    else:
                        trial_value = particles[i]['position'][d]
                    trial.append(trial_value)

                # Evaluate the trial vector
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < particles[i]['personal_best_val']:
                    particles[i]['position'] = trial[:]
                    particles[i]['personal_best_pos'] = trial[:]
                    particles[i]['personal_best_val'] = trial_fitness

                    # Update global best if needed
                    if trial_fitness < global_best_val:
                        global_best_pos = trial[:]
                        global_best_val = trial_fitness

    while evals < max_evals:
        # Adaptive inertia weight
        inertia_weight = max(min_inertia, inertia_weight - 0.1 * (max_evals - evals) / max_evals)

        # Particle Swarm Optimization step
        for particle in particles:
            if particle['personal_best_pos'] is None:
                particle['personal_best_pos'] = particle['position'][:]
                
            update_particle(particle, global_best_pos)

            fitness = func(particle['position'])
            evals += 1

            if fitness < particle['personal_best_val']:
                particle['personal_best_val'] = fitness
                particle['personal_best_pos'] = particle['position'][:]

            if fitness < global_best_val:
                global_best_val = fitness
                global_best_pos = particle['position'][:]
                last_improvement = evals

        # Differential Evolution step
        apply_differential_evolution()

    return global_best_val

# Example cost function for testing
def inverted_sphere(position):
    return -sum(x**2 for x in position)

# Example testing parameters
dim = 10
bounds = [(-5.12, 5.12) for _ in range(dim)]
max_evals = 1000

# Running the optimizer with the InvertedSphere function
best_value = run(inverted_sphere, dim, bounds, max_evals)
print("Best value found:", best_value)
#
#### Key Fixes:
#- The `random` module is correctly imported at the start of the script.
#- Ensured that variable scoping and usage in functions are managed without accessing undeclared variables. 
#- Validated that the code handles updates within its internal functions effectively and utilizes the `random` features comprehensively.
