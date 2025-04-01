#Here's an innovative implementation of a metaheuristic algorithm that combines elements of Simulated Annealing (SA) and Particle Swarm Optimization (PSO). The algorithm leverages the stochastic exploration of SA with the social and cognitive components of PSO to efficiently explore the solution space:
#
import random
import math

def run(func, dim, bounds, max_evals):
    
    # Initializing parameters
    num_particles = 30  # Number of particles in the swarm
    alpha = 0.9  # Cooling schedule parameter for SA
    temperature = 1.0  # Initial temperature for SA
    best = float('inf')
    evals = 0
    
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
            
            # Metropolis criterion for SA
            if random.random() < math.exp(-(fitness - particle['best_value']) / temperature):
                particle['best_position'] = particle['position'][:]
                particle['best_value'] = fitness
        
        # Update particle positions and velocities
        for particle in particles:
            if evals >= max_evals:
                break
            
            inertia = 0.5  # Inertia weight
            cognitive_component = 1.5  # Cognitive component coefficient
            social_component = 1.5  # Social component coefficient
            
            # Update velocity and position
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                cognitive_velocity = cognitive_component * r1 * (particle['best_position'][d] - particle['position'][d])
                social_velocity = social_component * r2 * (global_best_position[d] - particle['position'][d])
                particle['velocity'][d] = inertia * particle['velocity'][d] + cognitive_velocity + social_velocity
                
                # Update position
                particle['position'][d] += particle['velocity'][d]
                
                # Ensure position is within bounds
                if particle['position'][d] < bounds[d][0]:
                    particle['position'][d] = bounds[d][0]
                elif particle['position'][d] > bounds[d][1]:
                    particle['position'][d] = bounds[d][1]
        
        # Decrease temperature
        temperature *= alpha
    
    return global_best_value
#This code integrates a temperature-based exploration mechanism from SA with the search capabilities of PSO, allowing the swarm of particles to explore the search space more diversely and converge effectively on optimal solutions. Each iteration evaluates the fitness of solutions given by the particles, adjusting their velocities and positions as per both personal and global bests, while cooling the pseudo-temperature to allow for a controlled exploration-exploitation balance.
