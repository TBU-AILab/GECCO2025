import random

def run(func, dim, bounds, max_evals):
    # Initialize variables
    best = None
    best_params = None
    evals = 0

    # Define a basic mutation function
    def mutate(solution, bounds, mutation_rate=0.1):
        return [
            min(max(solution[i] + random.uniform(-mutation_rate, mutation_rate) * (bounds[i][1] - bounds[i][0]), bounds[i][0]), bounds[i][1])
            for i in range(dim)
        ]

    # Initialize a random solution
    current_solution = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
    current_fitness = func(current_solution)
    evals += 1
    best = current_fitness
    best_params = current_solution

    while evals < max_evals:
        # Mutate the current solution to explore the neighborhood
        candidate_solution = mutate(current_solution, bounds)
        candidate_fitness = func(candidate_solution)
        evals += 1

        # If the candidate is better, update current best
        if candidate_fitness < best:
            best = candidate_fitness
            best_params = candidate_solution

        # Adaptive mutation rate based on improvement
        mutation_rate = 0.1 * (evals / max_evals)

        # Introduce a greedy criteria for exploration
        if candidate_fitness < current_fitness:
            current_solution = candidate_solution
            current_fitness = candidate_fitness
        else:
            # Random exploration with decreased mutation rate
            if random.random() < mutation_rate:
                current_solution = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
                current_fitness = func(current_solution)
                evals += 1

    return best
#
