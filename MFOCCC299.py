import numpy as np
import matplotlib.pyplot as plt
import random


class CCCAOptimizer:
    """Candidates Cooperative Competitive Algorithm (CCCA) — single-task"""

    def __init__(self, bounds, pop_size=40, max_iter=100):
        self.bounds = bounds  # (lower, upper)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.theta = 0.6  # learning ability growth rate

    def initialize_population(self):
        """Initialize random population (students)"""
        lower, upper = self.bounds
        population = []
        for _ in range(self.pop_size):
            x = lower + (upper - lower) * random.random()
            population.append(x)
        return population

    def evaluate_fitness(self, x, objective_func):
        """Calculate fitness (score)"""
        return objective_func(x)

    def update_learning_ability(self, omega, iteration):
        """Update learning ability"""
        return 1 - omega * np.exp(-self.theta * iteration)

    def calculate_improvement_space(self, fitness_i, best_fitness, worst_fitness):
        """Normalized gap to best"""
        if abs(best_fitness - worst_fitness) < 1e-12:
            return 1.0
        return abs(best_fitness - fitness_i) / abs(best_fitness - worst_fitness)

    def self_study_update(self, student, teacher, omega, improvement_space):
        """Move toward teacher (no GA/PSO primitives)"""
        r1 = random.random()
        lower, upper = self.bounds
        new_student = student + r1 * omega * improvement_space * (teacher - student)
        return max(lower, min(upper, new_student))

    def one_on_one_assistance(self, weak_student, strong_student, omega, improvement_space):
        """Mentoring step for bottom half"""
        r2 = random.random()
        lower, upper = self.bounds
        new_student = weak_student + r2 * omega * improvement_space * (strong_student - weak_student)
        return max(lower, min(upper, new_student))

    def optimize(self, objective_func):
        """Main CCCA optimization loop (single-task)"""
        population = self.initialize_population()
        learning_abilities = [random.uniform(0, 1) for _ in range(self.pop_size)]

        best_fitness_history = []
        best_solution = None
        best_fitness = float('-inf')

        for iteration in range(self.max_iter):
            # Evaluate
            fitness_values = [self.evaluate_fitness(x, objective_func) for x in population]

            # Track best/worst
            current_best_idx = int(np.argmax(fitness_values))
            current_best_fitness = fitness_values[current_best_idx]
            current_best_solution = population[current_best_idx]
            worst_fitness = min(fitness_values)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution

            best_fitness_history.append(best_fitness)

            # Rank (best→worst)
            sorted_indices = sorted(range(len(fitness_values)),
                                    key=lambda i: fitness_values[i], reverse=True)

            new_population = []
            new_learning_abilities = []

            for i, idx in enumerate(sorted_indices):
                student = population[idx]
                omega = learning_abilities[idx]
                fitness_i = fitness_values[idx]

                improvement_space = self.calculate_improvement_space(
                    fitness_i, best_fitness, worst_fitness)

                # Self-study toward global best
                new_student = self.self_study_update(
                    student, best_solution, omega, improvement_space)

                # One-on-one assistance for bottom half
                if i >= self.pop_size // 2:
                    helper_idx = i - self.pop_size // 2  # paired with corresponding top half
                    if helper_idx < len(sorted_indices):
                        helper = population[sorted_indices[helper_idx]]
                        new_student = self.one_on_one_assistance(
                            new_student, helper, omega, improvement_space)

                new_population.append(new_student)

                # Update learning ability
                new_omega = self.update_learning_ability(omega, iteration)
                new_learning_abilities.append(new_omega)

            population = new_population
            learning_abilities = new_learning_abilities

        return best_solution, best_fitness, best_fitness_history


class MFCCCAOptimizer:
    """
    Integrated MFO + CCCA:
      • MFO provides unified population across tasks, skill factors, and cross-task transfer gate.
      • CCCA provides per-task update operators (self-study & one-on-one assistance),
        improvement-space scaling, and learning-ability schedule.
    """

    def __init__(self, tasks, pop_size=40, max_iter=100, theta=0.6, p_xfer=0.1):
        self.tasks = tasks                  # list of (bounds, objective_func)
        self.T = len(tasks)
        self.N = pop_size
        self.max_iter = max_iter
        self.theta = theta                  # CCCA ability schedule
        self.p_xfer = p_xfer               # cross-task transfer probability (MFO)

    # ---- CCCA primitives (used *inside* MFO update) ----
    def _self_study(self, x, teacher, omega, improv, bounds):
        r = random.random()
        new_x = x + r * omega * improv * (teacher - x)
        lo, hi = bounds
        return max(lo, min(hi, new_x))

    def _assist(self, weak, strong, omega, improv, bounds):
        r = random.random()
        new_x = weak + r * omega * improv * (strong - weak)
        lo, hi = bounds
        return max(lo, min(hi, new_x))

    def _improvement_space(self, f_i, f_best, f_worst):
        if abs(f_best - f_worst) < 1e-12:
            return 1.0
        return abs(f_best - f_i) / abs(f_best - f_worst)

    def _update_omega(self, omega, t):
        return 1 - omega * np.exp(-self.theta * t)

    # ---- MFO init: unified individual + skill factor ----
    def _init_population(self):
        pop, skill = [], []
        for _ in range(self.N):
            ind, fit = {}, {}
            for k, (bounds, obj) in enumerate(self.tasks):
                lo, hi = bounds
                x = lo + (hi - lo) * random.random()
                ind[k] = x
                fit[k] = obj(x)
            best_task = max(fit, key=fit.get)   # skill factor
            pop.append(ind)
            skill.append(best_task)
        return pop, skill, [random.random() for _ in range(self.N)]

    # ---- Integrated optimization ----
    def optimize(self):
        pop, skill, omega = self._init_population()

        # task-wise archives
        best_x = {k: None for k in range(self.T)}
        best_f = {k: float('-inf') for k in range(self.T)}
        hist   = {k: [] for k in range(self.T)}

        for t in range(self.max_iter):
            # evaluate all individuals on all tasks + refresh task archives
            F = []
            for i, ind in enumerate(pop):
                f_i = {}
                for k, (_, obj) in enumerate(self.tasks):
                    f = obj(ind[k])
                    f_i[k] = f
                    if f > best_f[k]:
                        best_f[k] = f
                        best_x[k] = ind[k]
                F.append(f_i)
            for k in range(self.T):
                hist[k].append(best_f[k])

            # per-task ranking & worst fitness (for assistance & improv-space)
            ranks = {k: sorted(range(self.N), key=lambda i: F[i][k], reverse=True)
                     for k in range(self.T)}
            worst_f = {k: F[ranks[k][-1]][k] for k in range(self.T)}

            # ----- Integration point: MFO variation uses CCCA operators per task -----
            new_pop, new_omega = [], []
            for i in range(self.N):
                ind_new = {}
                for k, (bounds, _) in enumerate(self.tasks):
                    x = pop[i][k]
                    f_i_k = F[i][k]

                    # choose teacher (usually same task; sometimes transfer from skilled task)
                    teacher_task = k
                    if random.random() < self.p_xfer and skill[i] != k and best_x[skill[i]] is not None:
                        teacher_task = skill[i]
                    teacher_x = best_x[teacher_task] if best_x[teacher_task] is not None else x

                    # CCCA self-study toward teacher
                    improv = self._improvement_space(f_i_k, best_f[k], worst_f[k])
                    x_new = self._self_study(x, teacher_x, omega[i], improv, bounds)

                    # CCCA one-on-one assistance for bottom half (per-task)
                    pos = ranks[k].index(i)
                    if pos >= self.N // 2:
                        helper_idx = ranks[k][pos - self.N // 2]
                        helper_x = pop[helper_idx][k]
                        x_new = self._assist(x_new, helper_x, omega[i], improv, bounds)

                    ind_new[k] = x_new

                new_pop.append(ind_new)
                new_omega.append(self._update_omega(omega[i], t))

            pop, omega = new_pop, new_omega

        return best_x, best_f, hist


def main():
    # Objective functions
    def f1(x):
        """f(x) = -(x-2)(x-6) = -x^2 + 8x - 12"""
        return -(x - 2) * (x - 6)

    def f2(x):
        """f(x) = -(x-200)(x-600) = -x^2 + 800x - 120000"""
        return -(x - 200) * (x - 600)

    print("=== CCCA Optimization Results (single-task runs) ===")

    # CCCA on Problem 1
    ccca1 = CCCAOptimizer(bounds=(0, 9), pop_size=30, max_iter=50)
    best_x1, best_f1, history1 = ccca1.optimize(f1)
    print(f"\nProblem 1: f(x) = -(x-2)(x-6) on [0,9]")
    print(f"Optimal x (CCCA): {best_x1:.6f}")
    print(f"Maximum f(x) (CCCA): {best_f1:.6f}")
    print(f"Theoretical optimum: x=4, f(x)=4")

    # CCCA on Problem 2
    ccca2 = CCCAOptimizer(bounds=(0, 900), pop_size=30, max_iter=50)
    best_x2, best_f2, history2 = ccca2.optimize(f2)
    print(f"\nProblem 2: f(x) = -(x-200)(x-600) on [0,900]")
    print(f"Optimal x (CCCA): {best_x2:.6f}")
    print(f"Maximum f(x) (CCCA): {best_f2:.6f}")
    print(f"Theoretical optimum: x=400, f(x)=40000")

    print("\n=== Integrated MF-CCCA Results (MFO + CCCA) ===")

    # Solve both problems simultaneously with MF-CCCA
    tasks = [
        ((0, 9), f1),      # Task 0
        ((0, 900), f2)     # Task 1
    ]
    mfccca = MFCCCAOptimizer(tasks, pop_size=40, max_iter=50, theta=0.6, p_xfer=0.1)
    mf_solutions, mf_fitness, mf_history = mfccca.optimize()

    print(f"\nMF-CCCA - Problem 1: f(x) = -(x-2)(x-6)")
    print(f"Optimal x: {mf_solutions[0]:.6f}")
    print(f"Maximum f(x): {mf_fitness[0]:.6f}")

    print(f"\nMF-CCCA - Problem 2: f(x) = -(x-200)(x-600)")
    print(f"Optimal x: {mf_solutions[1]:.6f}")
    print(f"Maximum f(x): {mf_fitness[1]:.6f}")

    # --------- Plots ---------
    plt.figure(figsize=(12, 8))

    # Convergence Problem 1
    plt.subplot(2, 2, 1)
    plt.plot(history1, label='CCCA (single-task)', linewidth=2)
    plt.plot(mf_history[0], 'r--', label='MF-CCCA (multi-task)', linewidth=2)
    plt.axhline(y=4, linestyle=':', label='Theoretical Max = 4')
    plt.title('Problem 1: f(x) = -(x-2)(x-6)')
    plt.xlabel('Iteration'); plt.ylabel('Best Fitness'); plt.legend(); plt.grid(True)

    # Convergence Problem 2
    plt.subplot(2, 2, 2)
    plt.plot(history2, label='CCCA (single-task)', linewidth=2)
    plt.plot(mf_history[1], 'r--', label='MF-CCCA (multi-task)', linewidth=2)
    plt.axhline(y=40000, linestyle=':', label='Theoretical Max = 40000')
    plt.title('Problem 2: f(x) = -(x-200)(x-600)')
    plt.xlabel('Iteration'); plt.ylabel('Best Fitness'); plt.legend(); plt.grid(True)

    # Landscape 1
    plt.subplot(2, 2, 3)
    x1_vals = np.linspace(0, 9, 200)
    y1_vals = [f1(x) for x in x1_vals]
    plt.plot(x1_vals, y1_vals, linewidth=2)
    plt.axvline(x=best_x1, linestyle='--', label=f'CCCA: x={best_x1:.2f}')
    plt.axvline(x=mf_solutions[0], color='orange', linestyle='--', label=f'MF-CCCA: x={mf_solutions[0]:.2f}')
    plt.axvline(x=4, linestyle=':', label='True optimum: x=4')
    plt.title('Function 1 Landscape'); plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(); plt.grid(True)

    # Landscape 2
    plt.subplot(2, 2, 4)
    x2_vals = np.linspace(0, 900, 200)
    y2_vals = [f2(x) for x in x2_vals]
    plt.plot(x2_vals, y2_vals, linewidth=2)
    plt.axvline(x=best_x2, linestyle='--', label=f'CCCA: x={best_x2:.0f}')
    plt.axvline(x=mf_solutions[1], color='orange', linestyle='--', label=f'MF-CCCA: x={mf_solutions[1]:.0f}')
    plt.axvline(x=400, linestyle=':', label='True optimum: x=400')
    plt.title('Function 2 Landscape'); plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --------- Simple error report ---------
    print("\n=== Analysis (absolute error in x) ===")
    print(f"Problem 1 - CCCA error:   {abs(best_x1 - 4):.6f}")
    print(f"Problem 1 - MF-CCCA error:{abs(mf_solutions[0] - 4):.6f}")
    print(f"Problem 2 - CCCA error:   {abs(best_x2 - 400):.6f}")
    print(f"Problem 2 - MF-CCCA error:{abs(mf_solutions[1] - 400):.6f}")


if __name__ == "__main__":
    main()
