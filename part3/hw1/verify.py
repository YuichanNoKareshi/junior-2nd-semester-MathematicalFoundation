import numpy as np
import math


def gaussian(x, mean, sigma, step):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma) * step


def integral(mean, sigma):
    x1 = mean - 3 * sigma
    x2 = mean + 3 * sigma
    ret = 0

    i = x1
    step = 0.01
    while i < x2:
        ret += gaussian(i, mean, sigma, step)
        i += step

    return ret



