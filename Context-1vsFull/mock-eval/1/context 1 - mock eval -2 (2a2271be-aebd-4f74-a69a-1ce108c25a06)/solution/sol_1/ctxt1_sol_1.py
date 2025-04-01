import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    evals = 0

    # Initialize a random solution within the bounds
    current_solution = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
    current_fitness = func(current_solution)
    evals += 1
    if current_fitness < best:
        best = current_fitness

    def neighborhood_search(point, scale=0.1):
        """Conducts a neighborhood search with adaptive scaling."""
        return [
            max(bounds[i][0], min(bounds[i][1], point[i] + random.uniform(-scale, scale)))
            for i in range(dim)
        ]

    # Algorithm body
    while evals < max_evals:
        # Introduce exploration-exploitation balance by adapting scale
        scale = 0.1 * (1 - evals / max_evals)

        new_solution = neighborhood_search(current_solution, scale)
        new_fitness = func(new_solution)
        evals += 1

        # If the new solution is better, accept it
        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness

            # Update the best found solution
            if new_fitness < best:
                best = new_fitness

    return best
