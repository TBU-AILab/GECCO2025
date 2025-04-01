import random

def run(func, dim, bounds, max_evals):
    best = float('inf')
    best_params = None
    
    def initialize_population(size, dim, bounds):
        return [[random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(size)]

    def evaluate_population(population):
        nonlocal eval_count
        fitness = []
        for individual in population:
            fitness_value = func(individual)
            eval_count += 1
            fitness.append((fitness_value, individual))
        return sorted(fitness, key=lambda x: x[0])
    
    def mutation(individual, bounds, mutation_rate=0.1):
        mutated = []
        for d in range(len(individual)):
            if random.random() < mutation_rate:
                mutated.append(random.uniform(bounds[d][0], bounds[d][1]))
            else:
                mutated.append(individual[d])
        return mutated
    
    def recombination(parent1, parent2):
        child = []
        for d in range(len(parent1)):
            if random.random() > 0.5:
                child.append(parent1[d])
            else:
                child.append(parent2[d])
        return child
    
    eval_count = 0
    population_size = 20
    population = initialize_population(population_size, dim, bounds)
    
    while eval_count < max_evals:
        evaluated_population = evaluate_population(population)
        if evaluated_population[0][0] < best:
            best, best_params = evaluated_population[0]
        
        next_generation = []
        
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(evaluated_population[:10], 2)
            child1 = recombination(parent1[1], parent2[1])
            child2 = recombination(parent2[1], parent1[1])
            child1 = mutation(child1, bounds)
            child2 = mutation(child2, bounds)
            
            next_generation.extend([child1, child2])

        population = next_generation
    
    return best
