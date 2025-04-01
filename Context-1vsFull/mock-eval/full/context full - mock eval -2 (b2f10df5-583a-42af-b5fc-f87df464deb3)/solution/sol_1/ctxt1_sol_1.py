#Here's an enhanced version of the previous implementation with some improvements, including adaptive parameters and more precise integration between the components of Simulated Annealing (SA) and Particle Swarm Optimization (PSO):
#
import random
import math

def run(func, dim, bounds, max_evals):
    
    # Initialize parameters
    num_particles = 40  # Increased number of particles
    alpha = 0.95  # Slow down the cooling for SA
    temperature = 1.0  # Initial temperature for SA
    evals = 0
    best = float('inf')
    
    # Particle initialization
    particles = [{'position': [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)],
                  'velocity': [0.0 for _ in range(dim)],
                  'best_position': None,
                  'best_value': float('inf')} for _ in range(num_particles)]
    
    global_best_position = None
    global_best_value = float('inf')
    
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
        
        # Update particle velocities and positions
        for particle in particles:
            if evals >= max_evals:
                break
            
            inertia = 0.7  # Inertia weight, updated for this implementation
            cognitive_component = 1.4  # Cognitive component coefficient
            social_component = 1.4  # Social component coefficient
            
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive_velocity = cognitive_component * r1 * (particle['best_position'][d] - particle['position'][d])
                social_velocity = social_component * r2 * (global_best_position[d] - particle['position'][d])
                particle['velocity'][d] = inertia * particle['velocity'][d] + cognitive_velocity + social_velocity
                
                # Metropolis criterion adaptation in velocity update
                if random.random() < math.exp(-abs(fitness - global_best_value) / temperature):
                    particle['velocity'][d] *= random.uniform(0.5, 1.5)
                
                # Update position
                particle['position'][d] += particle['velocity'][d]
                
                # Ensure position is within bounds
                if particle['position'][d] < bounds[d][0]:
                    particle['position'][d] = bounds[d][0]
                elif particle['position'][d] > bounds[d][1]:
                    particle['position'][d] = bounds[d][1]
        
        # Decrease temperature gradually
        temperature *= alpha
    
    return global_best_value
#
#Improvements made:
#- The number of particles was increased for better exploration.
#- The cooling rate `alpha` was adjusted for more gradual cooling.
#- The inertia, cognitive, and social components were fine-tuned to balance exploration and exploitation.
#- The Metropolis criterion's influence was integrated into the velocity update, introducing more randomness and adaptability in movement.
#- Enhanced boundary checks ensure particles remain within feasible regions.
#
#This implementation should provide robust performance across various types of objective functions due to its adaptive search strategy.
