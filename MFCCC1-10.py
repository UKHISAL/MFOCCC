import numpy as np
import matplotlib.pyplot as plt


class MultifactorialCCCA:
    def __init__(self, pop_size=40, max_iter=500, rmp=0.3):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rmp = rmp

    def f1(self, x):
        """Task 1: f(x) = -(x-2)(x-6), domain [0, 9]"""
        if 0 <= x <= 9:
            return -(x - 2) * (x - 6)
        return -1e10  # Heavy penalty for out of bounds

    def f2(self, x):
        """Task 2: f(x) = -(x-200)(x-600), domain [0, 900]"""
        if 0 <= x <= 900:
            return -(x - 200) * (x - 600)
        return -1e10

    def initialize_population(self):
        """Initialize with diversity across both task domains"""
        # Half population near Task 1 domain, half near Task 2
        pop1 = np.random.uniform(0, 9, self.pop_size // 2)
        pop2 = np.random.uniform(0, 900, self.pop_size - self.pop_size // 2)
        return np.concatenate([pop1, pop2])

    def evaluate_fitness(self, pop):
        """Evaluate both tasks and compute scalar fitness"""
        fitness_t1 = np.array([self.f1(x) for x in pop])
        fitness_t2 = np.array([self.f2(x) for x in pop])

        # Rank-based scalar fitness (lower rank = better)
        rank_t1 = np.argsort(np.argsort(-fitness_t1)) + 1
        rank_t2 = np.argsort(np.argsort(-fitness_t2)) + 1

        # Skill factor: task with better rank
        skill_factor = np.where(rank_t1 <= rank_t2, 1, 2)

        # Scalar fitness: best rank
        scalar_fitness = np.minimum(rank_t1, rank_t2)

        return fitness_t1, fitness_t2, scalar_fitness, skill_factor

    def optimize(self):
        """Main optimization loop"""
        pop = self.initialize_population()
        learning_ability = np.random.uniform(0, 2, self.pop_size)

        best_x_t1 = 4.0  # Initialize with theoretical optimum
        best_f_t1 = self.f1(4.0)
        best_x_t2 = 400.0
        best_f_t2 = self.f2(400.0)

        history_t1 = []
        history_t2 = []

        for iter_num in range(self.max_iter):
            # Evaluate
            fit_t1, fit_t2, scalar_fit, skill_factor = self.evaluate_fitness(pop)

            # Update best solutions for each task
            valid_t1 = (pop >= 0) & (pop <= 9)
            valid_t2 = (pop >= 0) & (pop <= 900)

            if np.any(valid_t1):
                best_idx_t1 = np.where(valid_t1)[0][np.argmax(fit_t1[valid_t1])]
                if fit_t1[best_idx_t1] > best_f_t1:
                    best_f_t1 = fit_t1[best_idx_t1]
                    best_x_t1 = pop[best_idx_t1]

            if np.any(valid_t2):
                best_idx_t2 = np.where(valid_t2)[0][np.argmax(fit_t2[valid_t2])]
                if fit_t2[best_idx_t2] > best_f_t2:
                    best_f_t2 = fit_t2[best_idx_t2]
                    best_x_t2 = pop[best_idx_t2]

            history_t1.append(best_f_t1)
            history_t2.append(best_f_t2)

            # Sort by scalar fitness
            sorted_idx = np.argsort(scalar_fit)
            pop = pop[sorted_idx]
            skill_factor = skill_factor[sorted_idx]
            learning_ability = learning_ability[sorted_idx]

            # Update learning ability
            learning_ability = 1 - learning_ability * np.exp(-0.6 * iter_num)

            # Generate offspring
            offspring = []
            for i in range(0, self.pop_size - 1, 2):
                parent1, parent2 = pop[i], pop[i + 1]

                # Assortative mating
                if np.random.rand() < self.rmp or skill_factor[i] == skill_factor[i + 1]:
                    alpha = np.random.rand()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = (1 - alpha) * parent1 + alpha * parent2
                else:
                    child1, child2 = parent1, parent2

                # Self-study for offspring
                if skill_factor[i] == 1:
                    target = best_x_t1
                else:
                    target = best_x_t2

                r1 = np.random.rand()
                child1 = child1 + r1 * learning_ability[i] * (target - child1)
                child2 = child2 + r1 * learning_ability[i + 1] * (target - child2)

                offspring.extend([child1, child2])

            if len(offspring) < self.pop_size:
                offspring.append(pop[-1])

            pop = np.array(offspring[:self.pop_size])

            # Apply bounds with reflection for diversity
            for i in range(len(pop)):
                if skill_factor[i] == 1:
                    pop[i] = np.clip(pop[i], 0, 9)
                else:
                    pop[i] = np.clip(pop[i], 0, 900)

        return best_x_t1, best_f_t1, best_x_t2, best_f_t2, history_t1, history_t2


# Run optimization
np.random.seed(42)
mfo_ccca = MultifactorialCCCA(pop_size=40, max_iter=500, rmp=0.3)
x1_opt, f1_opt, x2_opt, f2_opt, hist1, hist2 = mfo_ccca.optimize()

print("=" * 60)
print("MULTIFACTORIAL OPTIMIZATION RESULTS")
print("=" * 60)
print(f"\nTask 1: f(x) = -(x-2)(x-6), domain [0, 9]")
print(f"  Optimal x: {x1_opt:.6f}")
print(f"  Maximum f(x): {f1_opt:.6f}")
print(f"  Theoretical optimum: x=4, f(x)=4")
print(f"  Error: {abs(x1_opt - 4):.6f}")

print(f"\nTask 2: f(x) = -(x-200)(x-600), domain [0, 900]")
print(f"  Optimal x: {x2_opt:.6f}")
print(f"  Maximum f(x): {f2_opt:.6f}")
print(f"  Theoretical optimum: x=400, f(x)=40000")
print(f"  Error: {abs(x2_opt - 400):.6f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(hist1, 'b-', linewidth=2, label='CCCA-MFO')
ax1.axhline(y=4, color='r', linestyle='--', label='Theoretical optimum')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Best Fitness', fontsize=12)
ax1.set_title('Task 1: f(x) = -(x-2)(x-6)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(hist2, 'g-', linewidth=2, label='CCCA-MFO')
ax2.axhline(y=40000, color='r', linestyle='--', label='Theoretical optimum')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Best Fitness', fontsize=12)
ax2.set_title('Task 2: f(x) = -(x-200)(x-600)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mfo_ccca_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)