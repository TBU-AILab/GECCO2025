import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    evals = 0

    def neighborhood_search(point):
        delta = 0.1
        return [
            max(bounds[i][0], min(bounds[i][1], point[i] + random.uniform(-delta, delta)))
            for i in range(dim)
        ]

    # Initialize with a random solution
    current_solution = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]
    current_fitness = func(current_solution)
    evals += 1
    if current_fitness < best:
        best = current_fitness

    # Algorithm body
    while evals < max_evals:
        new_solution = neighborhood_search(current_solution)
        new_fitness = func(new_solution)
        evals += 1

        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness

            if new_fitness < best:
                best = new_fitness
    
    return best
