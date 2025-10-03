import numpy as np
import random


# Function definitions
def f1(x):
    return -(x - 2) * (x - 6)


def f2(x):
    return -(x - 200) * (x - 600)


# Helper functions for Factorial Rank, Skill Factor, Scalar Fitness
def factorial_rank(population, task):
    """
    Rank the population based on fitness for a given task.
    """
    fitness = [task(ind) for ind in population]
    ranked = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
    return ranked


def skill_factor(ranked1, ranked2, population):
    """
    Calculate the skill factor for each individual based on task performance.
    """
    skill_factors = []
    for ind in population:
        # Find the rank of the individual in both ranked lists
        rank1 = ranked1.index(population.index(ind))
        rank2 = ranked2.index(population.index(ind))

        # Skill factor: Choose the task with the best rank (lowest rank)
        skill_factor = 0 if rank1 < rank2 else 1  # 0 for task 1, 1 for task 2
        skill_factors.append(skill_factor)
    return skill_factors


def scalar_fitness(ranked1, ranked2, skill_factors):
    """
    Calculate the scalar fitness based on factorial ranks and skill factors.
    """
    fitness = []
    for i in range(len(skill_factors)):
        # Use the rank of the best task for each individual
        best_rank = ranked1 if skill_factors[i] == 0 else ranked2
        scalar_fitness = 1 / (best_rank.index(i) + 1)  # Scalar fitness = 1 / rank
        fitness.append(scalar_fitness)
    return fitness


# Initialize population
def initialize_population(pop_size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(pop_size)]


# Selection and Elitism (keep the best individuals)
def select_population(population, fitness, pop_size):
    sorted_pop = [x for _, x in sorted(zip(fitness, population), reverse=True)]
    return sorted_pop[:pop_size]


# Main optimization loop
def optimize_mfo_ccc(pop_size, lower_bound, upper_bound, iterations=8, rmp=0.5):
    # Initialize population
    population = initialize_population(pop_size, lower_bound, upper_bound)

    for iteration in range(iterations):
        # Evaluate fitness for both tasks
        ranked1 = factorial_rank(population, f1)
        ranked2 = factorial_rank(population, f2)

        # Calculate skill factor for each individual
        skill_factors = skill_factor(ranked1, ranked2, population)

        # Calculate scalar fitness for each individual
        fitness = scalar_fitness(ranked1, ranked2, skill_factors)

        # Print iteration-wise values for better visibility
        print(f"Iteration {iteration + 1}:")
        print(f"Population: {population}")
        print(f"Fitness: {fitness}")
        print(f"Best Individual: {population[0]}")
        print("-" * 50)

        # Selection: Keep the best individuals
        population = select_population(population, fitness, pop_size)

        # Apply the cooperation and competition mechanisms
        # Cooperation - Share information between good performers
        best_individual = population[0]
        for i in range(1, len(population)):
            # Cooperation updates based on sharing information with a random perturbation (rmp)
            if random.random() < rmp:  # If the random value is below rmp, cooperate
                population[i] = (population[i] + best_individual) / 2 + random.uniform(-0.1, 0.1)
            else:  # Else, retain some individuality
                population[i] = population[i] + random.uniform(-0.1, 0.1)

        # Competition - Eliminate poor performers
        population = select_population(population, fitness, pop_size)

    # Final result
    best_individual = population[0]
    return best_individual


# Run optimization with unified population space [0, 900]
pop_size = 10
lower_bound, upper_bound = 0, 900

optimal_x = optimize_mfo_ccc(pop_size, lower_bound, upper_bound)
print("Optimal X found:", optimal_x)
