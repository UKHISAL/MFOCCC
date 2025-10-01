import numpy as np
import matplotlib.pyplot as plt
import random


class CCCAOptimizer:
    """Candidates Cooperative Competitive Algorithm (CCCA)"""

    def __init__(self, bounds, pop_size=40, max_iter=100):
        self.bounds = bounds  # (lower, upper)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.theta = 0.6  # Learning ability growth rate

    def initialize_population(self):
        """Initialize random population (students)"""
        lower, upper = self.bounds
        population = []
        for _ in range(self.pop_size):
            x = lower + (upper - lower) * random.random()
            population.append(x)
        return population

    def evaluate_fitness(self, x, objective_func):
        """Calculate fitness (total score)"""
        return objective_func(x)

    def update_learning_ability(self, omega, iteration):
        """Update learning ability using Eq. (4)"""
        return 1 - omega * np.exp(-self.theta * iteration)

    def calculate_improvement_space(self, fitness_i, best_fitness, worst_fitness):
        """Calculate improvement space using Eq. (5)"""
        if abs(best_fitness - worst_fitness) < 1e-10:
            return 1.0
        return abs(best_fitness - fitness_i) / abs(best_fitness - worst_fitness)

    def self_study_update(self, student, best_student, omega, improvement_space):
        """the core of ccc self-study update using Eq. (3) """
        r1 = random.random()
        lower, upper = self.bounds

        # Update towards best student
        new_student = r1 * omega * improvement_space * (best_student - student) + student

        # Keep within bounds
        new_student = max(lower, min(upper, new_student))
        return new_student

    def one_on_one_assistance(self, weak_student, strong_student, omega, improvement_space):
        """second step of ccc on en one assistance using Eq. (6)"""
        r2 = random.random()
        lower, upper = self.bounds

        new_student = r2 * omega * improvement_space * (strong_student - weak_student) + weak_student
        new_student = max(lower, min(upper, new_student))
        return new_student

    def optimize(self, objective_func):
        """Main CCCA optimization loop The main CCCA optimization loop integrates
        both the self-study and one-on-one assistance operations as part of its
        population update process. The population is updated using
        these core CCCA principles, alongside the learning ability that evolves over time."""
        # Initialize population
        population = self.initialize_population()
        learning_abilities = [random.uniform(0, 1) for _ in range(self.pop_size)]

        best_fitness_history = []
        best_solution = None
        best_fitness = float('-inf')

        for iteration in range(self.max_iter):
            # Evaluate fitness for all students
            fitness_values = [self.evaluate_fitness(x, objective_func) for x in population]

            # Find best and worst
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_solution = population[current_best_idx]

            worst_fitness = min(fitness_values)

            # Update global best
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution

            best_fitness_history.append(best_fitness)

            # Sort population by fitness (best to worst)
            sorted_indices = sorted(range(len(fitness_values)),
                                    key=lambda i: fitness_values[i], reverse=True)

            new_population = []
            new_learning_abilities = []

            for i, idx in enumerate(sorted_indices):
                student = population[idx]
                omega = learning_abilities[idx]
                fitness_i = fitness_values[idx]

                # Calculate improvement space
                improvement_space = self.calculate_improvement_space(
                    fitness_i, best_fitness, worst_fitness)

                # Self-study update (everyone learns from the best)
                new_student = self.self_study_update(
                    student, best_solution, omega, improvement_space)

                # One-on-one assistance (bottom 50% get help from top 50%)
                if i >= self.pop_size // 2:  # Bottom half
                    helper_idx = i - self.pop_size // 2  # Paired with top half
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


class MFOOptimizer:
    """Multi-Factorial Optimization for handling multiple tasks simultaneously"""

    def __init__(self, tasks, pop_size=40, max_iter=100):
        self.tasks = tasks  # List of (bounds, objective_func) tuples
        self.num_tasks = len(tasks)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.theta = 0.6

    def initialize_multifactorial_population(self):
        """Initialize population that can solve multiple tasks he population
        is initialized for both tasks at the same time. Each
        individual in the population holds a solution for both tasks (Task 1 and Task 2)."""
        population = []
        skill_factors = []

        # Which task each individual is best at

        for i in range(self.pop_size):
            individual = {}
            individual_fitness = {}

            # Evaluate individual on all tasks
            for task_id, (bounds, obj_func) in enumerate(self.tasks):
                lower, upper = bounds
                x = lower + (upper - lower) * random.random()
                individual[task_id] = x
                individual_fitness[task_id] = obj_func(x)

            # Determine skill factor (best task for this individual)
            best_task = max(individual_fitness.keys(),
                            key=lambda k: individual_fitness[k])

            population.append(individual)
            skill_factors.append(best_task)

        return population, skill_factors

    def optimize_multifactorial(self):
        """Multi-factorial optimization"""
        population, skill_factors = self.initialize_multifactorial_population()
        learning_abilities = [random.uniform(0, 1) for _ in range(self.pop_size)]

        # Track best solutions for each task
        task_best_solutions = {}
        task_best_fitness = {}
        task_fitness_history = {i: [] for i in range(self.num_tasks)}

        for task_id in range(self.num_tasks):
            task_best_solutions[task_id] = None
            task_best_fitness[task_id] = float('-inf')

        for iteration in range(self.max_iter):
            # Evaluate all individuals on all tasks Evaluation of Fitness for Both Tasks:
            # During the optimization loop, each individual is evaluated on both tasks in parallel:
            all_fitness = []
            for i, individual in enumerate(population):
                fitness_dict = {}
                for task_id, (bounds, obj_func) in enumerate(self.tasks):
                    fitness_dict[task_id] = obj_func(individual[task_id])
                all_fitness.append(fitness_dict)

            # Update best solutions for each task
            for task_id in range(self.num_tasks):
                for i, fitness_dict in enumerate(all_fitness):
                    if fitness_dict[task_id] > task_best_fitness[task_id]:
                        task_best_fitness[task_id] = fitness_dict[task_id]
                        task_best_solutions[task_id] = population[i][task_id]

                task_fitness_history[task_id].append(task_best_fitness[task_id])

            # Update population using multifactorial approach
            new_population = []
            new_learning_abilities = []

            for i in range(self.pop_size):
                individual = population[i]
                omega = learning_abilities[i]
                skill_factor = skill_factors[i]

                new_individual = {}

                # Update for each task dimension
                for task_id, (bounds, obj_func) in enumerate(self.tasks):
                    lower, upper = bounds
                    current_x = individual[task_id]

                    # Use best solution from the individual's skill factor task
                    # but also learn from other tasks occasionally
                    #Each individual has a skill factor that indicates which task they are best at. When updating solutions
                    # for each individual, the algorithm occasionally allows individuals to learn from tasks they are not best at.
                    if task_id == skill_factor or random.random() < 0.1:
                        best_x = task_best_solutions[task_id]
                        if best_x is not None:
                            r1 = random.random()
                            improvement_space = 0.5  # Simplified
                            new_x = r1 * omega * improvement_space * (best_x - current_x) + current_x
                            new_x = max(lower, min(upper, new_x))
                            new_individual[task_id] = new_x
                        else:
                            new_individual[task_id] = current_x
                    else:
                        # Keep current value or small random perturbation
                        perturbation = (upper - lower) * 0.01 * (random.random() - 0.5)
                        new_x = current_x + perturbation
                        new_x = max(lower, min(upper, new_x))
                        new_individual[task_id] = new_x

                new_population.append(new_individual)

                # Update learning ability
                new_omega = 1 - omega * np.exp(-self.theta * iteration)
                new_learning_abilities.append(new_omega)

            population = new_population
            learning_abilities = new_learning_abilities

        return task_best_solutions, task_best_fitness, task_fitness_history


def main():
    # Define objective functions
    def f1(x):
        """f(x) = -(x-2)(x-6) = -(x^2 - 8x + 12) = -x^2 + 8x - 12"""
        return -(x - 2) * (x - 6)

    def f2(x):
        """f(x) = -(x-200)(x-600) = -(x^2 - 800x + 120000) = -x^2 + 800x - 120000"""
        return -(x - 200) * (x - 600)

    print("=== CCCA Optimization Results ===")

    # Solve first problem with CCCA
    ccca1 = CCCAOptimizer(bounds=(0, 9), pop_size=30, max_iter=50)
    best_x1, best_f1, history1 = ccca1.optimize(f1)

    print(f"\nProblem 1: f(x) = -(x-2)(x-6) on [0,9]")
    print(f"Optimal x: {best_x1:.6f}")
    print(f"Maximum f(x): {best_f1:.6f}")
    print(f"Theoretical optimum: x=4, f(x)=4")

    # Solve second problem with CCCA
    ccca2 = CCCAOptimizer(bounds=(0, 900), pop_size=30, max_iter=50)
    best_x2, best_f2, history2 = ccca2.optimize(f2)

    print(f"\nProblem 2: f(x) = -(x-200)(x-600) on [0,900]")
    print(f"Optimal x: {best_x2:.6f}")
    print(f"Maximum f(x): {best_f2:.6f}")
    print(f"Theoretical optimum: x=400, f(x)=40000")

    print("\n=== Multi-Factorial Optimization (MFO) Results ===")

    # Solve both problems simultaneously with MFO
    tasks = [
        ((0, 9), f1),  # Task 0: first problem
        ((0, 900), f2)  # Task 1: second problem
    ]

    mfo = MFOOptimizer(tasks, pop_size=40, max_iter=50)
    mfo_solutions, mfo_fitness, mfo_history = mfo.optimize_multifactorial()

    print(f"\nMFO - Problem 1: f(x) = -(x-2)(x-6)")
    print(f"Optimal x: {mfo_solutions[0]:.6f}")
    print(f"Maximum f(x): {mfo_fitness[0]:.6f}")

    print(f"\nMFO - Problem 2: f(x) = -(x-200)(x-600)")
    print(f"Optimal x: {mfo_solutions[1]:.6f}")
    print(f"Maximum f(x): {mfo_fitness[1]:.6f}")

    # Plot convergence curves
    plt.figure(figsize=(12, 8))

    # Plot for Problem 1
    plt.subplot(2, 2, 1)
    plt.plot(history1, 'b-', label='CCCA', linewidth=2)
    plt.plot(mfo_history[0], 'r--', label='MFO', linewidth=2)
    plt.axhline(y=4, color='g', linestyle=':', label='Theoretical Max = 4')
    plt.title('Problem 1: f(x) = -(x-2)(x-6)')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)

    # Plot for Problem 2
    plt.subplot(2, 2, 2)
    plt.plot(history2, 'b-', label='CCCA', linewidth=2)
    plt.plot(mfo_history[1], 'r--', label='MFO', linewidth=2)
    plt.axhline(y=40000, color='g', linestyle=':', label='Theoretical Max = 40000')
    plt.title('Problem 2: f(x) = -(x-200)(x-600)')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)

    # Plot function landscapes
    plt.subplot(2, 2, 3)
    x1_vals = np.linspace(0, 9, 100)
    y1_vals = [f1(x) for x in x1_vals]
    plt.plot(x1_vals, y1_vals, 'b-', linewidth=2)
    plt.axvline(x=best_x1, color='r', linestyle='--', label=f'CCCA: x={best_x1:.2f}')
    plt.axvline(x=mfo_solutions[0], color='orange', linestyle='--', label=f'MFO: x={mfo_solutions[0]:.2f}')
    plt.axvline(x=4, color='g', linestyle=':', label='True optimum: x=4')
    plt.title('Function 1 Landscape')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    x2_vals = np.linspace(0, 900, 100)
    y2_vals = [f2(x) for x in x2_vals]
    plt.plot(x2_vals, y2_vals, 'b-', linewidth=2)
    plt.axvline(x=best_x2, color='r', linestyle='--', label=f'CCCA: x={best_x2:.0f}')
    plt.axvline(x=mfo_solutions[1], color='orange', linestyle='--', label=f'MFO: x={mfo_solutions[1]:.0f}')
    plt.axvline(x=400, color='g', linestyle=':', label='True optimum: x=400')
    plt.title('Function 2 Landscape')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n=== Analysis ===")
    print(f"Problem 1 - CCCA Error: {abs(best_x1 - 4):.6f}")
    print(f"Problem 1 - MFO Error: {abs(mfo_solutions[0] - 4):.6f}")
    print(f"Problem 2 - CCCA Error: {abs(best_x2 - 400):.6f}")
    print(f"Problem 2 - MFO Error: {abs(mfo_solutions[1] - 400):.6f}")


if __name__ == "__main__":
    main()