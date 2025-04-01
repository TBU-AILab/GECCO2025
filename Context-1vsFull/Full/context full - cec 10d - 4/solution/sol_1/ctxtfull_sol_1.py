import random

def run(func, dim, bounds, max_evals):
    def generate_solution():
        """Generate a potential solution within the bounds."""
        return [random.uniform(b[0], b[1]) for b in bounds]
    
    def mutate_solution(solution, scale=0.1):
        """Mutate the solution slightly by a random factor."""
        return [max(min(x + random.uniform(-scale, scale), bounds[i][1]), bounds[i][0]) for i, x in enumerate(solution)]
    
    def crossover(parent1, parent2):
        """Perform crossover between two solutions to create a child."""
        return [random.choice(pair) for pair in zip(parent1, parent2)]

    def selection(pool, fitness_pool):
        """Select two parents using tournament selection."""
        tournament_size = 3
        selected = random.sample(range(len(pool)), tournament_size)
        selected.sort(key=lambda i: fitness_pool[i])
        return pool[selected[0]], pool[selected[1]]

    # Initialize a population of solutions
    population_size = 20
    population = [generate_solution() for _ in range(population_size)]
    fitness_population = [func(indiv) for indiv in population]
    evals = population_size  # Count the initial evaluations

    # Record the best solution
    best_solution = min(zip(population, fitness_population), key=lambda x: x[1])
    best_fitness = best_solution[1]

    while evals < max_evals:
        new_population = []
        new_fitness = []

        for _ in range(population_size):
            # Select parents
            parent1, parent2 = selection(population, fitness_population)

            # Crossover and possibly mutate
            child = crossover(parent1, parent2)
            if random.random() < 0.3:
                child = mutate_solution(child)

            # Evaluate new child solution
            child_fitness = func(child)
            evals += 1

            # Add to the new population
            new_population.append(child)
            new_fitness.append(child_fitness)

            # Check for a new best solution
            if child_fitness < best_fitness:
                best_solution = child
                best_fitness = child_fitness

        # Replace the old population with the new one
        population = new_population
        fitness_population = new_fitness

    # Return the fitness of the best found solution
    return best_fitness
#
#This updated algorithm employs a simple evolutionary strategy with a tournament selection mechanism for parent selection, crossover for new solution creation, and mutation for exploration. The population-based approach enhances explorative capabilities, ensuring that diverse regions of the search space are covered, potentially improving average outcomes over a series of runs. The consideration of a small population size maintains computational feasibility within constraint evaluation limits.
