import math
import numpy as np


def run(func, dim, bounds, max_evals):
    best_solution = None
    best_score = float('inf')

    for i in range(max_evals):
        candidate_solution = [np.random.uniform(low, high) for low, high in bounds]
        candidate_score = func(candidate_solution)

        if candidate_score < best_score:
            best_score = candidate_score
            best_solution = candidate_solution


    return best_score
