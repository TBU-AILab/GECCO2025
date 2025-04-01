#Certainly! Let's correct that issue. The `nonlocal` keyword is incorrectly used because there is no outer (non-global) scope where `inertia_weight` is defined. It should just be accessed directly as an attribute of the class. Here's the corrected code:
#
import random

class MetaheuristicOptimizer:
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        
        # Parameters for Particle Swarm Optimization
        self.num_particles = 50
        self.inertia_weight = 0.7
        self.min_inertia = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

        # Parameters for Differential Evolution
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

        # Initialize swarm with particles
        self.particles = [{'position': [random.uniform(low, high) for low, high in bounds],
                           'velocity': [0.0] * dim,
                           'personal_best_pos': None,
                           'personal_best_val': float('inf')} for _ in range(self.num_particles)]

        self.global_best_pos = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        self.global_best_val = float('inf')
        self.evals = 0

    def update_particle(self, particle):
        # Particle Swarm Optimization update
        for d in range(self.dim):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.cognitive_coeff * r1 * (particle['personal_best_pos'][d] - particle['position'][d])
            social_velocity = self.social_coeff * r2 * (self.global_best_pos[d] - particle['position'][d])
            particle['velocity'][d] = self.inertia_weight * particle['velocity'][d] + cognitive_velocity + social_velocity

            # Update position
            particle['position'][d] += particle['velocity'][d]

            # Boundary constraints
            particle['position'][d] = min(max(particle['position'][d], self.bounds[d][0]), self.bounds[d][1])

    def apply_differential_evolution(self):
        for i in range(self.num_particles):
            if random.random() < self.crossover_prob:
                a, b, c = random.sample([j for j in range(self.num_particles) if j != i], 3)
                trial = []

                for d in range(self.dim):
                    if random.random() < 0.5:
                        trial_value = self.particles[a]['position'][d] + self.mutation_factor * (self.particles[b]['position'][d] - self.particles[c]['position'][d])
                        # Ensure within bounds
                        trial_value = max(min(trial_value, self.bounds[d][1]), self.bounds[d][0])
                    else:
                        trial_value = self.particles[i]['position'][d]
                    trial.append(trial_value)

                # Evaluate the trial vector
                trial_fitness = self.func(trial)
                self.evals += 1

                if trial_fitness < self.particles[i]['personal_best_val']:
                    self.particles[i]['position'] = trial[:]
                    self.particles[i]['personal_best_pos'] = trial[:]
                    self.particles[i]['personal_best_val'] = trial_fitness

                    # Update global best if needed
                    if trial_fitness < self.global_best_val:
                        self.global_best_pos = trial[:]
                        self.global_best_val = trial_fitness

    def run(self):
        last_improvement = 0
        
        while self.evals < self.max_evals:
            # Adaptive inertia weight
            self.inertia_weight = max(self.min_inertia, self.inertia_weight - 0.1 * (self.max_evals - self.evals) / self.max_evals)

            # Particle Swarm Optimization step
            for particle in self.particles:
                if particle['personal_best_pos'] is None:
                    particle['personal_best_pos'] = particle['position'][:]
                    
                self.update_particle(particle)

                fitness = self.func(particle['position'])
                self.evals += 1

                if fitness < particle['personal_best_val']:
                    particle['personal_best_val'] = fitness
                    particle['personal_best_pos'] = particle['position'][:]

                if fitness < self.global_best_val:
                    self.global_best_val = fitness
                    self.global_best_pos = particle['position'][:]
                    last_improvement = self.evals

            # Differential Evolution step
            self.apply_differential_evolution()

        return self.global_best_val

# Example cost function for testing
def inverted_sphere(position):
    return -sum(x**2 for x in position)

# Example testing parameters
dim = 10
bounds = [(-5.12, 5.12) for _ in range(dim)]
max_evals = 1000

# Running the optimizer with the InvertedSphere function
optimizer = MetaheuristicOptimizer(inverted_sphere, dim, bounds, max_evals)
best_value = optimizer.run()
print("Best value found:", best_value)
#
#### Key Changes:
#- The `nonlocal` keyword has been removed from `update_particle` method as it's unnecessary. Now, the method accesses `inertia_weight` directly from the instance.
#- All method calls and attribute accesses should now be correctly scoped to the instance of the class.
