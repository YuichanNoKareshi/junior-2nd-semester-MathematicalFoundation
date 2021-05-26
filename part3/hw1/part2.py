import numpy as np
import math
import matplotlib.pyplot as plt


def do2_1():
    means = [0, 1]  # 均值μ
    sigmas = [math.sqrt(1), math.sqrt(4)]  # 标准差δ

    for i in range(0, 2):
        for j in range(0, 2):
            x = np.linspace(means[i] - 3 * sigmas[j], means[i] + 3 * sigmas[j], 50)  # 定义域
            y = np.exp(-(x - means[i]) ** 2 / (2 * sigmas[j] ** 2)) / (math.sqrt(2 * math.pi) * sigmas[j])  # 定义曲线函数
            plt.plot(x, y, "g", linewidth=2)  # 加载曲线

    plt.grid(True)  # 网格线
    plt.show()  # 显示


def gaussian(x, mean, sigma, step):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma) * step


def integral(mean, sigma):
    x1 = mean - 3 * sigma
    x2 = mean + 3 * sigma
    result = 0

    i = x1
    step = 0.01
    while i < x2:
        result += gaussian(i, mean, sigma, step)
        i += step

    return result


def do2_2():
    means = [0, 1]  # 均值μ
    sigmas = [math.sqrt(1), math.sqrt(4)]  # 标准差δ

    print("part2_2: the integrals of these 4 Gaussian functions are")
    for i in range(0, 2):
        for j in range(0, 2):
            print(integral(means[i], sigmas[j]))
    print("\n")


