import random

def knapsack_evolution(pop, values, weights, W, generations=100):
    parent_percent = 0.2  # Top % of the population selected as parents
    mutation_chance = 0.08  # Probability of mutation
    pop_size = len(pop)
    
    def fitness(individual):
        """Calculate total value with penalty for exceeding weight."""
        total_value = sum(v for v, selected in zip(values, individual) if selected)
        total_weight = sum(w for w, selected in zip(weights, individual) if selected)
        penalty = max(0, total_weight - W) * 10  # Penalty for exceeding weight
        return total_value - penalty

    for _ in range(generations):
        # **Selection**: Rank population by fitness
        pop.sort(key=fitness, reverse=True)
        parent_length = int(parent_percent * len(pop))
        parents = pop[:parent_length]

        # **Crossover**: Generate children
        children = []
        while len(children) < pop_size - parent_length:
            male = random.choice(parents)
            female = random.choice(parents)
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)

        # **Mutation**: Randomly flip bits in children
        for child in children:
            if random.random() < mutation_chance:
                r = random.randint(0, len(child) - 1)
                child[r] = 1 - child[r]  # Flip 0 to 1 or 1 to 0

        # New generation: Parents + Children
        pop = parents + children

    # Return the best individual
    best_solution = max(pop, key=fitness)
    return best_solution, fitness(best_solution)

# Example Usage
values = [60, 100, 120]  # Item values
weights = [10, 20, 30]   # Item weights
W = 50  # Knapsack capacity

# Initialize a random population of binary solutions
pop_size = 10
pop = [[random.choice([0, 1]) for _ in range(len(values))] for _ in range(pop_size)]

best_solution, best_value = knapsack_evolution(pop, values, weights, W)
print(f"Best Solution: {best_solution}")
print(f"Best Value: {best_value}")
