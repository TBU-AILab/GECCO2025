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
    
    best_solution = generate_solution()
    best_fitness = func(best_solution)
    evals = 1  # Count the initial evaluation

    while evals < max_evals:
        # Randomly select two solutions and create a child solution
        solution1 = generate_solution()
        solution2 = generate_solution()
        child = crossover(solution1, solution2)

        # Optionally mutate the child solution
        if random.random() < 0.5:
            child = mutate_solution(child)

        # Evaluate new child solution
        child_fitness = func(child)
        evals += 1

        # Check if the new child is a better solution
        if child_fitness < best_fitness:
            best_solution = child
            best_fitness = child_fitness

    # Return the fitness of the best found solution
    return best_fitness
#
#This code implements a simple hybrid metaheuristic where solutions are generated, evaluated, and potentially improved through crossover and mutation strategies. It carefully counts the evaluations and searches for minimal values as directed. The code is arranged to ensure functionality within the bounds and constraints of the given framework.
