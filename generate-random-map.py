import itertools
import random

import numpy as np
from typing import List, Tuple
from collections.abc import Sequence, Iterator
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

def getInsertionIndex(value: float, cumsum: Sequence[float]) -> Tuple[int, float]:
    right = np.searchsorted(cumsum, value, side='right')
    beforeVal = cumsum[right-1] if right > 0 else 0
    afterVal = cumsum[right]
    rest = (value - beforeVal)/(afterVal - beforeVal)
    return right, rest


def generate_single_index(cum_density_map: Sequence[float]) -> Tuple[int, float]:
    S = cum_density_map[-1]
    v = S * random.random()
    i = getInsertionIndex(v, cum_density_map)
    return i


def generate_random_indices(count: int, cum_density_map: Sequence[float]) -> Iterator[Tuple[int, float]]:
    for _ in itertools.repeat(None, count):
        yield generate_single_index(cum_density_map)


def unravel_index(indices: Sequence[Tuple[int, float]], shape: Tuple[int, int]) -> Iterator[List[float]]:
    for ind in indices:
        val, rest = ind
        o = np.unravel_index(val, shape)
        yield tuple(t+random.random() for t in reversed(o))




def gen_random_map(max_count: int, density: npt.ArrayLike) -> Iterator[tuple[float, float]]:
    cum_sum = np.cumsum(density, dtype=float)
    shape = np.shape(density)
    indices = list(generate_random_indices(max_count, cum_sum))
    print("indices: ", indices)
    return unravel_index(indices, shape)



if __name__ == "__main__":
    def square(i, maxN):
        return -(i-maxN/2)**2 + (maxN/2)**2

    d = np.fromfunction(lambda j, i: square(i, 200)*square(j, 150), (150, 200), dtype=float)

    img = image.imread('test-bw.bmp')
    print(img.shape)
    print(img.dtype)
    # print(img)
    img_normalized = np.divide(img, 256.)
    img_hsv = matplotlib.colors.rgb_to_hsv(img_normalized)
    img_sat = img_hsv[:, :, 1]
    img_hue = img_hsv[:, :, 2]
    d = img_hue


    print(d)
    M = d  # np.array([[1, 1, 2, 0], [0, 3, 4, 5]])
    t = list(gen_random_map(8*24, M))
    v = np.array(t)
    S = np.shape(M)
    plt.axes(xlim=(0, S[1]), ylim=(0, S[0]))
    plt.scatter(v[:,0], v[:,1], s=1)
    plt.show()
