#To further enhance the optimization algorithm, we can improve the diversity management and introduce a more sophisticated mutation strategy. Additionally, we can refine boundary handling slightly and tweak the adaptive inertia and parameter adjustments for improved convergence rates. Here is an improved version of the metaheuristic algorithm:
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

    def velocity_clamp(v, min_v, max_v):
        """Clamp velocity to be within the specified bounds"""
        return max(min_v, min(v, max_v))
    
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
        cognitive_component = 1.5 - 0.5 * (evals / max_evals)  # Dynamic decrease
        social_component = 1.5 + 0.5 * (evals / max_evals)    # Dynamic increase

        diversity_threshold = 0.1 * sum(bounds[d][1] - bounds[d][0] for d in range(dim)) / dim
        mutation_intensity = 0.1

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
                    new_velocity += random.uniform(-0.1, 0.1)
                
                # Apply velocity clamp
                particle['velocity'][d] = velocity_clamp(new_velocity, -1, 1)

                # Update position
                particle['position'][d] += particle['velocity'][d]

                # Ensure position is within bounds
                particle['position'][d] = max(min(particle['position'][d], bounds[d][1]), bounds[d][0])
        
        # Enhanced diversity management
        if evals > 0 and evals % (max_evals // 5) == 0:
            # Calculate diversity using average distance
            distances = []
            for particle in particles:
                distances.append(sum(abs(p - g) for p, g in zip(particle['position'], global_best_position)))
            diversity = sum(distances) / len(distances)
            
            # Apply mutation if diversity is low
            if diversity < diversity_threshold:
                for particle in particles:
                    if random.random() < mutation_intensity:
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
#### Enhancements Made:
#1. **Dynamic Parameter Adjustment**: Using adaptive inertia, cognitive, and social parameters that adjust smoothly over evaluations for improved adaptability.
#2. **Effective Diversity Measurement**: Using average particle distance from the global best to trigger diversification strategies.
#3. **Diversity Threshold**: Dynamically determined based on the dimensionality and range of bounds, enabling consistent adaptation across different problem scales.
#4. **Sophisticated Mutation Strategy**: Featuring both random position resets and partial position mutations which are dependent on current diversity levels. 
#
#This implementation focuses on maintaining equilibrium between exploration and exploitation, effectively responding to diversity needs, and ensuring robust search behavior across complex solution spaces.
