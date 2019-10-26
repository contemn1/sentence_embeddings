import math
from functools import partial

import numpy as np


def dct_formula(N, k, index):
    return np.sqrt(2.0 / N) * np.cos(np.pi / N * (index + 0.5) * k)


def get_dct_coefficient(k, N):
    """
    :type k: int
    :param k:
    :return: 
    """
    assert k >= 0
    c_zero = np.full((1, N), math.sqrt(1.0 / N))
    if k == 0:
        return c_zero
    dct_func = partial(dct_formula, N)
    result = np.fromfunction(dct_func, (k, N))  # type: np.array
    result[0] = c_zero
    if k >= N:
        result[N:] = np.zeros(N)
    return result


def power_mean(p):
    def calculate_mean(array):
        if p == 1:
            return np.mean(array, axis=0)

        mean = np.power(array, p).mean(axis=0).astype('complex')
        return np.power(mean, (1.0 / p)).real

    return calculate_mean
