#To further refine the algorithm, we can enhance its adaptiveness and robustness by incorporating a more sophisticated diversity measure, improving boundary handling strategies, and introducing a more systematic mutation process when diversity drops. Here’s an updated version with these enhancements:
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
        cognitive_component = 1.0 + 1.0 * (1 - evals / max_evals)
        social_component = 1.0 + 1.0 * (evals / max_evals)

        diversity_threshold = 0.5
        mutation_intensity = 0.2

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
        
        # Enhanced diversity preservation
        if evals > 0 and evals % (max_evals // 5) == 0:
            # Calculate diversity
            avg_position = [sum(particle['position'][d] for particle in particles) / num_particles for d in range(dim)]
            diversity = sum(math.sqrt(sum((particle['position'][d] - avg_position[d]) ** 2 for d in range(dim)))
                            for particle in particles) / num_particles
            
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
#### Key Enhancements:
#1. **Enhanced Diversity Measurement**: Utilizes root-mean-square deviation for more representative diversity assessment.
#2. **Systematic Mutation Approach**: Conditional based on the diversity measure and is performed with a defined intensity.
#3. **More Robust Boundary Handling**: Ensures particles stay within bounds after positions are updated by mutations or velocity adjustments.
#4. **Integrated Mutation with Fitness Re-evaluation**: Mutated particles immediately update their fitness and potentially the global best.
#
#These refinements aim to optimize the balance between global exploration and local exploitation while maintaining diversity and robustness in the search process, especially in complex landscapes.
