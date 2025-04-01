import random

class Evaluator:
    def __init__(self, func):
        self.func = func
        self.evals = 0
    
    def evaluate(self, solution):
        self.evals += 1
        return self.func(solution)


class MetaheuristicOptimizer:
    def __init__(self, evaluator, dimensionality, bounds, max_evals):
        self.evaluator = evaluator
        self.dimensionality = dimensionality
        self.bounds = bounds
        self.max_evals = max_evals

    def random_solution(self):
        return [random.uniform(low, high) for low, high in self.bounds]

    def perturb_solution(self, solution, strength):
        return [
            max(self.bounds[d][0], min(self.bounds[d][1], solution[d] + strength * (random.random() * 2 - 1)))
            for d in range(self.dimensionality)
        ]

    def select_best(self, population):
        sorted_population = sorted(population, key=lambda x: x[1])
        return sorted_population[0]

    def optimize(self):
        population_size = 20
        perturbation_strength = 0.1
        
        # Initialize population with random solutions and evaluate them
        population = [(self.random_solution(), None) for _ in range(population_size)]
        population = [(indv, self.evaluator.evaluate(indv)) for indv, _ in population]
        
        best_solution, best_fitness = self.select_best(population)
        
        while self.evaluator.evals < self.max_evals:
            offspring = []
            
            for individual, fitness in population:
                # Perturb current individual to generate a new solution
                new_solution = self.perturb_solution(individual, perturbation_strength)
                new_fitness = self.evaluator.evaluate(new_solution)
                
                # Select the better solution
                if new_fitness < fitness:
                    offspring.append((new_solution, new_fitness))
                else:
                    offspring.append((individual, fitness))
            
            # Sort by fitness and retain the best individuals
            population = sorted(offspring, key=lambda x: x[1])[:population_size]
            current_best_solution, current_best_fitness = self.select_best(population)
            
            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best_solution, current_best_fitness

        return best_solution, best_fitness


def example_cost_function(x):
    """Example objective function to be minimized."""
    return sum(x_i**2 for x_i in x)

def run(func, dimensionality, bounds, max_evaluations):
    evaluator = Evaluator(func)
    optimizer = MetaheuristicOptimizer(evaluator, dimensionality, bounds, max_evaluations)
    best_solution, best_fitness = optimizer.optimize()
    print("Best Solution Found:", best_solution)
    print("Best Fitness Found:", best_fitness)
    print("Total Function Evaluations:", evaluator.evals)


if __name__ == "__main__":
    dimensionality = 5
    bounds = [(-5, 5) for _ in range(dimensionality)]
    max_evaluations = 100
    
    run(example_cost_function, dimensionality, bounds, max_evaluations)
