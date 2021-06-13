import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x * x * x * x - 29 * x * x * x * x / 20 + 29 * x * x * x / 36 - 31 * x * x / 144 + x / 36 - 1 / 720


def df(x):
    return 5 * x * x * x * x - 29 * x * x * x / 5 + 29 * x * x / 12 - 31 * x / 72 + 1 / 36


def plot_px():
    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    ax.plot(x, f(x), label='px')
    plt.grid(True)
    plt.show()


def do4_1():
    plot_px()


def Newton(x0):
    flag = 1
    x_curr = x0
    x_next = x0
    while flag == 1:
        x_next = x_curr - f(x_curr) / df(x_curr)
        if int(x_next * (10 ** 10)) == int(x_curr * (10 ** 10)):
            flag = 0
        x_curr = x_next

    return x_curr


def do4_2():
    root = Newton(0.45)
    print("Newton\n", root)


def Secant(x_1, x0):
    flag = 1
    xa = x_1
    xb = x0

    root_cur = xa - f(xa) * (xb - xa) / (f(xb) - f(xa))
    if f(xa) * f(root_cur) < 0:
        xb = root_cur
    if f(xb) * f(root_cur) < 0:
        xa = root_cur
    root_pre = root_cur

    while flag == 1:
        root_cur = xa - f(xa) * (xb - xa) / (f(xb) - f(xa))

        if int(root_pre * (10 ** 10)) == int(root_cur * (10 ** 10)):
            flag = 0

        if f(xa) * f(root_cur) < 0:
            xb = root_cur
        if f(xb) * f(root_cur) < 0:
            xa = root_cur

        root_pre = root_cur

    return root_cur


def do4_3():
    root = Secant(0, 0.18)
    print("Secant\n", root)
    print("x=0\n", f(0))
    print("x=0.18\n", f(0.18))
