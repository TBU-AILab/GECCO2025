import random

def run(func, dim, bounds, max_evals):
    def generate_solution():
        """Generate a potential solution within the bounds."""
        return [random.uniform(b[0], b[1]) for b in bounds]
    
    def differential_evolution_mutation(target_idx, scale=0.8):
        """Differential evolution mutation strategy."""
        indices = list(range(population_size))
        indices.remove(target_idx)
        a, b, c = random.sample(indices, 3)
        mutant = [
            max(min(population[a][i] + scale * (population[b][i] - population[c][i]), bounds[i][1]), bounds[i][0])
            for i in range(dim)
        ]
        return mutant
    
    def crossover(target, mutant, cr=0.9):
        """Perform binomial crossover to create trial vector."""
        return [
            mutant[i] if random.random() < cr else target[i]
            for i in range(dim)
        ]

    # Dynamic adaptation of population size
    population_size = max(4, min(100, dim * 10))
    population = [generate_solution() for _ in range(population_size)]
    fitness_population = [func(indiv) for indiv in population]
    evals = population_size  # Initial evaluations

    # Record the best solution
    best_idx = min(range(population_size), key=lambda i: fitness_population[i])
    best_solution = population[best_idx]
    best_fitness = fitness_population[best_idx]

    while evals < max_evals:
        for i in range(population_size):
            # Generate a mutant solution
            mutant = differential_evolution_mutation(i)
            
            # Crossover the target and mutant to produce a trial vector
            trial = crossover(population[i], mutant)
            
            # Evaluate trial solution
            trial_fitness = func(trial)
            evals += 1
            
            # Selection: greedy replacement
            if trial_fitness < fitness_population[i]:
                population[i] = trial
                fitness_population[i] = trial_fitness
                
                # Update best solution if the trial is better
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

    # Return the fitness of the best found solution
    return best_fitness
#
#This improved version of the differential evolution-based algorithm introduces a dynamic adaptation for population size. The population size is set between a minimum of 4 and a maximum of 100, determined proportionally to the problem dimension. This aims to adaptively balance exploration and exploitation efficiency given the problem's scope. This approach allows the algorithm to explore more thoroughly in larger spaces while keeping computational resources in check for smaller dimensions.
