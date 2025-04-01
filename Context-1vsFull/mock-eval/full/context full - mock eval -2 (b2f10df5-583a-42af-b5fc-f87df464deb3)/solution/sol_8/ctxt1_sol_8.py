#To further improve the solution, let's refine the algorithm with an enhanced balance of exploration and exploitation. The changes will include a more adaptive approach to parameter adjustments, a refined mutation mechanism, and improved particle position handling to ensure particles remain efficient and bounded. Here's an upgraded version:
#
import random
import math

def run(func, dim, bounds, max_evals):
    
    # Initialize parameters
    num_particles = 40
    initial_inertia = 0.9
    final_inertia = 0.4
    evals = 0
    
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

    def clamp(value, min_value, max_value):
        """Ensure a value is within a specified range."""
        return max(min(value, max_value), min_value)
    
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
        mutation_rate = 0.2 * (1 - evals / max_evals)

        # Update velocities and positions
        for particle in particles:
            if evals >= max_evals:
                break
            
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive_velocity = cognitive_component * r1 * (particle['best_position'][d] - particle['position'][d])
                social_velocity = social_component * r2 * (global_best_position[d] - particle['position'][d])
                particle['velocity'][d] = inertia * particle['velocity'][d] + cognitive_velocity + social_velocity

                # Apply velocity dampening
                particle['velocity'][d] *= (0.9 + 0.1 * random.random())
                
                # Ensure the velocity is within bounds
                particle['velocity'][d] = clamp(particle['velocity'][d], -abs(bounds[d][1] - bounds[d][0]), abs(bounds[d][1] - bounds[d][0]))

                # Update position
                particle['position'][d] += particle['velocity'][d]

                # Ensure position is within bounds
                particle['position'][d] = clamp(particle['position'][d], bounds[d][0], bounds[d][1])
        
        # Enhanced diversity management
        if evals > 0 and evals % (max_evals // 5) == 0:
            # Calculate diversity using average particle movement
            avg_position = [
                sum(particle['position'][d] for particle in particles) / num_particles
                for d in range(dim)
            ]
            diversity = math.sqrt(sum(
                (particle['position'][d] - avg_position[d]) ** 2
                for particle in particles for d in range(dim)
            ) / num_particles)
            
            # Apply mutation if diversity is low
            if diversity < diversity_threshold:
                for particle in particles:
                    if random.random() < mutation_rate:
                        # Targeted mutation
                        particle['position'] = [
                            random.uniform(bounds[d][0], bounds[d][1]) if random.random() < 0.1 else particle['position'][d]
                            for d in range(dim)
                        ]
                        # Re-evaluate fitness of mutated particle
                        fitness = func(particle['position'])
                        evals += 1
                        if fitness < particle['best_value']:
                            particle['best_value'] = fitness
                            particle['best_position'] = particle['position'][:]
                        if fitness < global_best_value:
                            global_best_value = fitness
                            global_best_position = particle['position'][:]
    
    return global_best_value
#
#### Key Enhancements:
#1. **Velocity Dampening**: Stability is increased by scaling the velocity with a dampening factor.
#2. **Improved Position and Velocity Handling**: Ensures particles and velocities stay within feasible bounds using a general clamp function.
#3. **Dynamic Mutation Strategy**: Introduces a systematic restart mechanism to maintain diversity, relying on the dynamic tracker of diversity below a threshold.
#4. **Contextual Parameter Adjustment**: Continues adapting cognitive and social components, ensuring stable convergence characteristics throughout the search process. 
#
#These improvements are tailored for enhanced control over particle behavior, maintaining diversity, and combating premature convergence, thereby fostering an efficiently optimized search process.
