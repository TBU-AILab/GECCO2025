#To further improve the algorithm, let's incorporate more sophisticated strategies for maintaining diversity and convergence. This includes using an adaptive mutation rate, a more targeted diversification strategy, and employing a velocity dampening mechanism to prevent overshooting. Here's the enhanced version of the algorithm:
#
import random
import math

def run(func, dim, bounds, max_evals):
    
    # Initialize parameters
    num_particles = 40
    initial_inertia = 0.9
    final_inertia = 0.4
    temperature = 1.0
    alpha = 0.97
    evals = 0
    best = float('inf')
    
    # Particle initialization
    particles = [{
        'position': [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)],
        'velocity': [random.uniform(-1, 1) for _ in range(dim)],
        'best_position': None,
        'best_value': float('inf')
    } for _ in range(num_particles)]
    
    global_best_position = None
    global_best_value = float('inf')
    
    def adaptive_inertia(evals, max_evals):
        """Adaptive inertia weight"""
        return initial_inertia - (initial_inertia - final_inertia) * (evals / max_evals)

    while evals < max_evals:
        for particle in particles:
            # Evaluate fitness
            fitness = func(particle['position'])
            evals += 1
            
            # Update personal best
            if fitness < particle['best_value']:
                particle['best_value'] = fitness
                particle['best_position'] = particle['position'][:]
            
            # Update global best
            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = particle['position'][:]
        
        inertia = adaptive_inertia(evals, max_evals)
        cognitive_component = 1.5 - 0.5 * (evals / max_evals)
        social_component = 1.5 + 0.5 * (evals / max_evals)

        diversity_threshold = 0.1 * sum(bounds[d][1] - bounds[d][0] for d in range(dim)) / dim
        mutation_intensity = 0.3 * (1 - evals / max_evals)  # Dynamic mutation intensity

        # Update velocities and positions
        for particle in particles:
            if evals >= max_evals:
                break
            
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive_velocity = cognitive_component * r1 * (particle['best_position'][d] - particle['position'][d])
                social_velocity = social_component * r2 * (global_best_position[d] - particle['position'][d])
                new_velocity = (inertia * particle['velocity'][d] + cognitive_velocity + social_velocity)

                # SA perturbation
                if random.random() < math.exp(-abs(fitness - global_best_value) / temperature):
                    new_velocity *= (0.9 + 0.2 * random.random())  # Velocity dampening

                # Apply velocity clamp
                particle['velocity'][d] = max(min(new_velocity, 1.0), -1.0)

                # Update position
                particle['position'][d] += particle['velocity'][d]

                # Ensure position is within bounds
                particle['position'][d] = max(min(particle['position'][d], bounds[d][1]), bounds[d][0])
        
        # Enhanced diversity management
        if evals > 0 and evals % (max_evals // 5) == 0:
            # Calculate diversity using average distance
            distances = [
                sum(abs(p - g) for p, g in zip(particle['position'], global_best_position))
                for particle in particles
            ]
            diversity = sum(distances) / len(distances)
            
            # Apply mutation if diversity is low
            if diversity < diversity_threshold:
                for particle in particles:
                    if random.random() < mutation_intensity:
                        # Targeted mutation with adaptive intensity
                        particle['position'] = [
                            random.uniform(bounds[d][0], bounds[d][1]) if random.random() < mutation_intensity else particle['position'][d]
                            for d in range(dim)
                        ]

                        # Re-evaluate mutated particle's fitness
                        fitness = func(particle['position'])
                        evals += 1
                        if fitness < particle['best_value']:
                            particle['best_value'] = fitness
                            particle['best_position'] = particle['position'][:]
                        if fitness < global_best_value:
                            global_best_value = fitness
                            global_best_position = particle['position'][:]
        
        # Gradual temperature decrease
        temperature *= alpha
    
    return global_best_value
#
#### Key Enhancements:
#1. **Adaptive Mutation Intensity**: Mutation intensity decreases over time, promoting exploration early and focusing on exploitation in later stages.
#2. **Velocity Dampening**: A dampening factor is applied during stochastic perturbations to control velocity magnitude.
#3. **Targeted Mutation Strategy**: Introduces mutations specifically when the measured diversity is below a threshold, focusing on exploration in critically low diversity scenarios.
#4. **Dynamic Parameter Adjustment**: Continued focus on tuning inertia, cognitive, and social factors based on evaluation progress.
#
#These improvements aim to boost the algorithm’s ability to maintain a balance between avoiding local optima and converging efficiently on global optima by dynamically adapting its strategies throughout its run.
