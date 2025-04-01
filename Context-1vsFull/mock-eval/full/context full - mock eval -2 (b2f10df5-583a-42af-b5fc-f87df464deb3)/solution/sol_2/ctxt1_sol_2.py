#To enhance the previous solution, let's incorporate adaptive mechanisms and a more refined control over exploration and exploitation balances. We'll add a dynamic adjustment for the cognitive and social components based on the iteration stage. Additionally, let's introduce random restarts to prevent premature convergence and improve exploration:
#
import random
import math

def run(func, dim, bounds, max_evals):
    
    # Initialize parameters
    num_particles = 40  # Number of particles
    initial_inertia = 0.9  # Starting inertia weight
    final_inertia = 0.4  # Ending inertia weight
    temperature = 1.0  # Initial temperature for SA
    alpha = 0.98  # Cooling schedule parameter for SA
    evals = 0
    best = float('inf')
    
    # Particle initialization
    particles = [{'position': [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)],
                  'velocity': [0.0 for _ in range(dim)],
                  'best_position': None,
                  'best_value': float('inf')} for _ in range(num_particles)]
    
    global_best_position = None
    global_best_value = float('inf')
    
    def adapt_inertia(t, max_evals):
        """ Adaptive inertia weight """
        return initial_inertia - (initial_inertia - final_inertia) * (t / max_evals)
    
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
        
        inertia = adapt_inertia(evals, max_evals)
        cognitive_component = 1.5 - 0.5 * (evals / max_evals)  # Decrease over time
        social_component = 1.5 + 0.5 * (evals / max_evals)  # Increase over time
        
        # Refresh positions and velocities
        for particle in particles:
            if evals >= max_evals:
                break

            for d in range(dim):
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive_velocity = cognitive_component * r1 * (particle['best_position'][d] - particle['position'][d])
                social_velocity = social_component * r2 * (global_best_position[d] - particle['position'][d])
                particle['velocity'][d] = inertia * particle['velocity'][d] + cognitive_velocity + social_velocity
                
                # SA perturbation in velocity update
                if random.random() < math.exp(-abs(fitness - global_best_value) / temperature):
                    particle['velocity'][d] *= random.uniform(0.8, 1.2)
                
                # Update position
                particle['position'][d] += particle['velocity'][d]
                
                # Ensure position is within bounds
                particle['position'][d] = max(min(particle['position'][d], bounds[d][1]), bounds[d][0])
        
        # Random restart to escape local optima
        if evals > 0 and evals % (max_evals // 5) == 0:
            for _ in range(num_particles // 5):
                i = random.randrange(num_particles)
                particles[i]['position'] = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        
        # Gradual temperature decrease
        temperature *= alpha
    
    return global_best_value
#
#### Key Improvements:
#1. **Adaptive Inertia**: The inertia weight decreases over time to transition from exploration to exploitation.
#2. **Adjusting Cognitive and Social Components**: These components dynamically change based on the iteration stage to emphasize individual learning early and global exploration later.
#3. **Simulated Annealing Influence**: The velocity update is subtly perturbed by a factor influenced by the Metropolis criterion.
#4. **Random Restart Strategy**: Occasionally reinitializing a subset of particles prevents stagnation and maintains diversity in the particle swarm. 
#
#These enhancements aim to maintain a balance between finding global optima and avoiding local traps efficiently.
