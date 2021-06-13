import numpy as np
from numpy import linalg as LA
import math


def f(n, x):
    return math.sqrt(2) * math.sin(np.pi * n * x)


def f2(n, x):
    return (f(n, x + 0.01) - 2 * f(n, x) + f(n, x - 0.01)) / (0.01 * 0.01)


x = np.arange(0.01, 1, 0.01)


def ex2_3():
    y = []
    for xi in x:
        y.append(f(1, xi))

    k2y = []
    for xi in x:
        k2y.append(-(np.pi * np.pi) * f(1, xi))

    h2M = []

    for i in range(1, 100):
        for j in range(1, 100):
            if i == j:
                h2M.append(-2 / (0.01 * 0.01))
            elif i - j == 1:
                h2M.append(1 / (0.01 * 0.01))
            elif j - i == 1:
                h2M.append(1 / (0.01 * 0.01))
            else:
                h2M.append(0)

    h2M = np.array([h2M])
    h2M = h2M.reshape(99, 99)

    My = h2M @ y

    bias = []

    for i in range(0, 99):
        bias.append(abs(My[i] - k2y[i]) / (abs(My[i] + k2y[i]) / 2))

    return My, k2y, bias, h2M


def do2_3():
    global My, h2M
    My, k2y, bias, h2M = ex2_3()
    print("bias\n", bias, '\n')


def do2_4():
    My = np.array(h2M)
    w, v = LA.eig(My)
    print("w\n", w)
    print("v\n", v, '\n')



