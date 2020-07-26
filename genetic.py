import numpy as np
from typing import Callable

class GeneticAlgorithm:
    def __init__(self,
                 population: list,
                 calculate_fitness: Callable[..., float],
                 mutate: Callable[],
                 crossover: Callable[]):
        self.calculate_fitness = calculate_fitness
        self.mutate = mutate
        self.crossover = crossover
        self.population = population
        self.population_size: int = len(population)
        self.pct_breed: float = 0.2
        self.pct_mutation: float = 0.02
        self.num_generations: int = 1
        self.rng = np.random.default_rng()
        self.best = None

    def __mutate(self, individual):
        if self.rng.random(1) < self.pct_mutation:
            return self.mutate(individual)
        else:
            return individual


    def evolve(self) -> None:
        for _ in range(self.num_generations):
            fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
            fitness_scores_np = np.array(fitness_scores)

            best_of_gen_ind = fitness_scores_np.argmax()
            if not self.best or fitness_scores_np[best_of_gen_ind] > self.best[1]:
                self.best = (self.population[best_of_gen_ind].copy(), fitness_scores_np[best_of_gen_ind])

            new_population = self.population.copy()
            weights = fitness_scores_np / fitness_scores_np.sum()
            num_breeds = int(self.pct_breed * self.population_size)

            for _ in range(num_breeds):
                parents = self.rng.choice(self.population, size=2, replace=False, p=weights)
                children = self.__mutate(self.crossover(parents[0], parents[1]))
                if type(children) == list:
                    new_population = new_population + children
                    fitness_scores = fitness_scores + [self.calculate_fitness(child) for child in children]
                else:
                    new_population.append(children)
                    fitness_scores.append(self.calculate_fitness(children))

            fittest = np.argpartition(fitness_scores, self.population_size)[-self.population_size:]
            self.population = list(map(new_population.__getitem__, fittest))

    def get_best(self):
        # This case is if evolve() was never called
        if not self.best:
            fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
            best_index = fitness_scores.index(max(fitness_scores))
            self.best = (self.population[best_index].copy(), fitness_scores[best_index])

        return self.best[0]