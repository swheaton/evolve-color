import numpy as np
import matplotlib.pyplot as plt

from skimage import measure

import genetic

HEIGHT = 200
WIDTH = 200

def random_image(height, width):
    random_gen = np.random.default_rng()
    random_img = random_gen.integers(0, 256, (height, width, 3))
    return random_img


# https://stackoverflow.com/questions/40919936/calculating-entropy-from-glcm-of-an-image
def fitness(img):
    shannon_entropy = (measure.shannon_entropy(img) - 1.584962500721156) ** 2
    return shannon_entropy


def crossover_image(img1, img2):
    random_gen = np.random.default_rng()
    img_mask = random_gen.integers(0, 1, img1.shape, endpoint=True)
    inv_img_mask = img_mask ^ 1
    np.repeat(img_mask[..., np.newaxis], repeats=3, axis=2)
    img1 *= img_mask
    img2 *= img_mask ^ 1
    return img1 * img_mask + img2 * inv_img_mask


def mutate_image(img):
    return img


population = [random_image(HEIGHT, WIDTH) for _ in range(20)]

ga = genetic.GeneticAlgorithm(population, fitness, crossover_image, mutate_image)
ga.evolve()
best_image = ga.get_best()

plt.imshow(best_image)
plt.show()

