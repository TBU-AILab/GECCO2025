import random

def run(func, dim, bounds, max_evals):
    evals = 0
    best_solution = None
    best_fitness = float('inf')
    
    # [Algorithm body: Hybrid Particle Swarm Optimization with Simulated Annealing]

    # Initialize particles
    particles = [{'position': [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)],
                  'velocity': [0.0 for _ in range(dim)],
                  'best_position': None,
                  'best_fitness': float('inf')}
                 for _ in range(10)]  # 10 particles as an example

    # Initialize global best
    global_best_position = None
    global_best_fitness = float('inf')

    # Begin iterations
    while evals < max_evals:
        for particle in particles:
            # Evaluate fitness
            current_fitness = func(particle['position'])
            evals += 1

            # Update personal best
            if current_fitness < particle['best_fitness']:
                particle['best_fitness'] = current_fitness
                particle['best_position'] = particle['position'][:]

            # Update global best
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle['position'][:]

            # Update best solution found
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = particle['position'][:]

            # Early stop if max_evals is reached
            if evals >= max_evals:
                break
        
        # Simulated Annealing-inspired velocity and position update
        temperature = max(1 - evals / max_evals, 0)
        for particle in particles:
            inertia = 0.5
            cognitive = 1.5 * random.random()
            social = 1.5 * random.random()

            for i in range(dim):
                # Update velocity
                inertia_comp = inertia * particle['velocity'][i]
                cognitive_comp = cognitive * (particle['best_position'][i] - particle['position'][i])
                social_comp = social * (global_best_position[i] - particle['position'][i])
               
                particle['velocity'][i] = inertia_comp + cognitive_comp + social_comp

                # Simulate annealing noise
                annealing_noise = (random.random() - 0.5) * temperature
                particle['position'][i] += particle['velocity'][i] + annealing_noise
                
                # Ensure particles remain within bounds
                if particle['position'][i] < bounds[i][0]:
                    particle['position'][i] = bounds[i][0]
                elif particle['position'][i] > bounds[i][1]:
                    particle['position'][i] = bounds[i][1]

    # Return the fitness of the best found solution
    return best_fitness
#This implementation combines elements of Particle Swarm Optimization (PSO) and Simulated Annealing. Each particle adjusts its position based on its own experience, the best global solution, and a small random noise, emulating the effect of annealing temperature that decreases over time. This hybrid approach aims to balance exploration and exploitation effectively while adhering to the constraint of maximum function evaluations.
