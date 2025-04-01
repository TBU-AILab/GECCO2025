import random

def run(func, dim, bounds, max_evals):
    # Initialize variables
    best_solution = [random.uniform(low, high) for low, high in bounds]
    best_fitness = func(best_solution)
    evals = 1

    # Adaptive step size for exploration and exploitation
    step_size = [(high - low) / 10.0 for low, high in bounds]

    # Incrementally updating neighborhood search based on feedback
    while evals < max_evals:
        # Generate a new candidate solution with slight disturbance
        candidate_solution = []
        for i in range(dim):
            # Small perturbations (Gaussian noise) while respecting bounds
            perturbation = random.gauss(0, step_size[i] / 3.0)
            candidate_value = best_solution[i] + perturbation
            candidate_value = min(max(candidate_value, bounds[i][0]), bounds[i][1])
            candidate_solution.append(candidate_value)

        # Evaluate the new candidate solution
        candidate_fitness = func(candidate_solution)
        evals += 1

        # Check and adapt the step size dynamically
        if candidate_fitness < best_fitness:
            # Improvement found, accept the candidate and decrease step size (exploitation)
            best_fitness = candidate_fitness
            best_solution = candidate_solution[:]
            step_size = [step * 0.9 for step in step_size]
        else:
            # No improvement, slightly increase step size (exploration)
            step_size = [step * 1.1 for step in step_size]

        # Apply a lower bound on the step size to maintain exploration capability
        step_size = [max(step, 0.001 * (high - low)) for step, (low, high) in zip(step_size, bounds)]

    # Return the fitness of the best solution found
    return best_fitness
#
#In this updated version of the adaptive search algorithm, the following improvements have been made:
#
#1. **Initialization with Evaluation**: The algorithm begins by evaluating an initial random solution to set a baseline for the best fitness and initializes it as the current best solution.
#
#2. **Gaussian Perturbations**: A Gaussian distribution is used to create small perturbations in each dimension to explore the neighborhood of the current best solution.
#
#3. **Dynamic Step Size Adjustments**: The algorithm dynamically adjusts the step size based on improvements. If an improvement is found, it reduces the step size to exploit the locality; otherwise, it increases the step size to explore further.
#
#4. **Step Size Bounds**: The algorithm ensures that the step sizes do not become too small, maintaining a minimum exploratory capacity by enforcing a lower bound based on the scale of the search space.
#
#These changes aim to improve exploration and exploitation balance, enhancing the optimizer's capability to seek an optimal solution effectively.
