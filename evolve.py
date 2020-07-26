import numpy as np
import matplotlib.pyplot as plt

import genetic


def random_image(height, width):
    random_gen = np.random.default_rng()
    random_img = random_gen.integers(0, 256, (height, width, 3))
    return random_img


def fitness(img):
    entropy = img.entropy()
    return entropy


def crossover_image(img1, img2):
    return img1


def mutate_image(img):
    return img


population = [random_image(200, 200) for _ in range(20)]

ga = genetic.GeneticAlgorithm(population, fitness, crossover_image, mutate_image)
ga.evolve()
best_image = ga.get_best()

ran = random_image(200, 200)
plt.imshow(ran)
plt.show()

