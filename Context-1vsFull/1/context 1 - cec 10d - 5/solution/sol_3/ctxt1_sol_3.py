#Based on this updated feedback, the current solution is already significantly better across all tested functions with strong statistical evidence. To further fortify the performance, if improvements are still sought, we can incorporate some advanced techniques generally used in the field of constrained optimization:
#
#1. **Adaptive Random Restart Mechanism**: This helps in escaping local minima by periodically restarting particles from different configurations.
#2. **Crowding Distance Mechanism**: Encourage diversity by using a crowding distance metric to prefer solutions that maintain diversity in the population.
#3. **Penalty Functions for Constraints**: If constraints are present in the problem, we can incorporate penalty functions to handle constraint violations.
#
#Since the given context indicates significant improvement across functions, we will implement an adaptive random restart mechanism to further strengthen exploration capabilities.
#
#Let's integrate an adaptive random restart mechanism into the current optimizer:
#
import random

def run(func, dim, bounds, max_evals):
    # Parameters
    num_particles = 50
    inertia_weight = 0.7
    min_inertia = 0.4
    cognitive_coeff = 1.5
    social_coeff = 1.5
    mutation_factor = 0.5
    crossover_prob = 0.9
    restart_prob = 0.1  # Probability of restart for adaptive random restart
    restart_threshold = 0.05  # Improvement threshold for considering a restart beneficial
    
    # Particle Swarm Optimization structure
    particles = [{'position': [random.uniform(low, high) for low, high in bounds],
                  'velocity': [0.0] * dim,
                  'best_pos': None,
                  'best_fitness': float('inf')} for _ in range(num_particles)]
    
    global_best_pos = None
    global_best_fitness = float('inf')
    
    evals = 0
    last_improvement = 0

    # Function for velocity and position update
    def update_velocity_and_position(particle):
        for d in range(dim):
            r1 = random.random()
            r2 = random.random()
            cognitive_term = cognitive_coeff * r1 * ((particle['best_pos'][d] if particle['best_pos'] else particle['position'][d]) - particle['position'][d])
            social_term = social_coeff * r2 * ((global_best_pos[d] if global_best_pos else particle['position'][d]) - particle['position'][d])
            particle['velocity'][d] = inertia_weight * particle['velocity'][d] + cognitive_term + social_term
            
            particle['position'][d] += particle['velocity'][d]

            # Boundary handling
            if particle['position'][d] < bounds[d][0]:
                particle['position'][d] = bounds[d][0]
                particle['velocity'][d] *= -0.5
            elif particle['position'][d] > bounds[d][1]:
                particle['position'][d] = bounds[d][1]
                particle['velocity'][d] *= -0.5

    # Adaptive random restart
    def adaptive_restart():
        nonlocal global_best_pos, global_best_fitness, last_improvement
        if evals - last_improvement > restart_threshold * max_evals:
            if random.random() < restart_prob:
                global_best_pos = None
                global_best_fitness = float('inf')
                last_improvement = evals
                # Reinitialize particles
                for particle in particles:
                    particle['position'] = [random.uniform(low, high) for low, high in bounds]
                    particle['velocity'] = [0.0] * dim
                    particle['best_pos'] = None
                    particle['best_fitness'] = float('inf')
                print("Adaptive Restart Triggered.")

    while evals < max_evals:
        inertia_weight = max(min_inertia, inertia_weight - 0.1 * (max_evals - evals) / max_evals)

        for particle in particles:
            update_velocity_and_position(particle)

            # Evaluate fitness
            fitness = func(particle['position'])
            evals += 1
            
            if fitness < particle['best_fitness']:
                particle['best_fitness'] = fitness
                particle['best_pos'] = particle['position'][:]
            
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_pos = particle['position'][:]
                last_improvement = evals

        adaptive_restart()
    
    return global_best_fitness
#
#This implementation includes an adaptive random restart mechanism designed to break out of local minima traps and enhance the exploration capabilities of the optimizer. This feature is particularly useful for maintaining and improving optimal performance across a wide range of problem instances.
