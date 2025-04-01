import random

def run(func, dim, bounds, max_evals):
    def generate_solution():
        """Generate a potential solution within the bounds."""
        return [random.uniform(b[0], b[1]) for b in bounds]
    
    def mutate_solution(solution, scale=0.1):
        """Mutate the solution slightly by a random factor."""
        return [max(min(x + random.uniform(-scale, scale), bounds[i][1]), bounds[i][0]) for i, x in enumerate(solution)]
    
    def polynomial_mutation(solution, mutation_rate=0.1, eta=20):
        """Apply polynomial mutation to introduce diversity."""
        for i in range(len(solution)):
            if random.random() < mutation_rate:
                delta = random.random()
                if delta < 0.5:
                    delta_q = (2 * delta) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
                solution[i] = max(min(solution[i] + delta_q, bounds[i][1]), bounds[i][0])
        return solution
    
    def crossover(parent1, parent2):
        """Perform crossover between two solutions to create a child."""
        alpha = random.random()
        return [alpha * x + (1 - alpha) * y for x, y in zip(parent1, parent2)]

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

            # Crossover the parents
            child = crossover(parent1, parent2)
            
            # Apply polynomial mutation for improved exploration
            if random.random() < 0.3:
                child = polynomial_mutation(child)

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
#I have added polynomial mutation to provide more diverse exploration of the solution space, which should help improve performance on more complex landscapes. This change is targeted at achieving better results on a wider array of functions by exploring more possibilities during mutation. The crossover mechanism retains blending between solutions using a weighted average for smoother combinations of parent characteristics.
