#Here is an improved version that incorporates a few more strategic changes to further enhance the balance between exploration and exploitation, while also integrating a diversity preservation mechanism:
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
        cognitive_component = 1.0 + 1.0 * (1 - evals / max_evals)  # Start high and decrease
        social_component = 1.0 + 1.0 * (evals / max_evals)  # Start low and increase

        diversity_threshold = 0.1  # Threshold for triggering diversity preservation
        mutation_rate = 0.1  # Rate of mutation during diversity preservation
        
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
                
                # Temporal perturbation based on SA
                if random.random() < math.exp(-abs(fitness - global_best_value) / temperature):
                    new_velocity += random.uniform(-0.1, 0.1)
                
                # Apply velocity clamp
                particle['velocity'][d] = velocity_clamp(new_velocity, -1, 1)

                # Update position
                particle['position'][d] += particle['velocity'][d]

                # Ensure position is within bounds
                particle['position'][d] = max(min(particle['position'][d], bounds[d][1]), bounds[d][0])
        
        # Diversity preservation mechanism
        if evals > 0 and evals % (max_evals // 5) == 0:
            # Measure diversity based on position variance
            positions = [particle['position'] for particle in particles]
            avg_position = [sum(p[d] for p in positions) / num_particles for d in range(dim)]
            diversity = math.sqrt(sum(
                (particle['position'][d] - avg_position[d]) ** 2 for particle in particles for d in range(dim)
            ) / num_particles)
            
            if diversity < diversity_threshold:
                # Introduce mutation
                for particle in particles:
                    if random.random() < mutation_rate:
                        particle['position'] = [
                            random.uniform(bounds[d][0], bounds[d][1]) if random.random() < 0.1 else particle['position'][d]
                            for d in range(dim)
                        ]
        
        # Slow down the cooling
        temperature *= alpha
    
    return global_best_value
#
#### Key Enhancements:
#1. **Velocity Initialization**: Particles now start with a small random velocity, adding initial exploration momentum.
#2. **Adaptive Inertia and Components**: Improved the adaptive components for inertia and cognitive/social factors for more articulated exploration-exploitation balance over time.
#3. **Velocity Clamping**: Ensures that velocities are contained within a defined range to prevent erratic movements.
#4. **Diversity Preservation**: Monitors particle diversity and introduces random mutations if diversity falls below a threshold, keeping the algorithm from premature convergence.
#5. **Controlled SA Perturbation**: Fine-tuned with a smaller range for perturbation to avoid large deviations while exploring.
#
#These improvements aim to maintain solution diversity while ensuring a consistent progress towards optimal solutions.
