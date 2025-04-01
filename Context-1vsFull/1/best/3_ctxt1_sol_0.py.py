import random

def run(func, dim, bounds, max_evals):
    # Initialize variables
    best_solution = None
    best_fitness = float('inf')
    evals = 0

    # Adaptive step size for exploration and exploitation
    step_size = [(high - low) / 10.0 for low, high in bounds]

    # Random initial solution
    current_solution = [random.uniform(low, high) for low, high in bounds]
    
    while evals < max_evals:
        # Evaluate current solution
        current_fitness = func(current_solution)
        evals += 1

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution[:]
        
        # Generate a new candidate solution
        candidate_solution = current_solution[:]
        for i in range(dim):
            if random.random() < 0.5:  # Random decision to change the parameter for diversification
                candidate_solution[i] += random.uniform(-step_size[i], step_size[i])
                # Ensure the candidate solution is within bounds
                candidate_solution[i] = min(max(candidate_solution[i], bounds[i][0]), bounds[i][1])

        # Evaluate the new candidate solution
        candidate_fitness = func(candidate_solution)
        evals += 1

        # Greedy acceptance criterion
        if candidate_fitness < current_fitness:
            current_solution = candidate_solution[:]
            current_fitness = candidate_fitness

        # Dynamically adjust step size based on progress
        step_size = [max(0.1 * (high - low), step * 0.9) for (low, high), step in zip(bounds, step_size)]

    # Return the fitness of the best solution found
    return best_fitness
#This implementation describes an adaptive random search algorithm tailored to perform both exploration and exploitation. The algorithm starts with a randomly generated solution within given bounds and makes use of an adaptive step size to ensure it efficiently searches the solution space. A greedy acceptance criterion is employed to select better solutions, and the step size adjusts dynamically based on search progress. It is designed to minimize the number of function evaluations while searching for an optimal solution within the constraints of the given bounds.
