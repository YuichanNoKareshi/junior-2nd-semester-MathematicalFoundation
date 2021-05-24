import math
import numpy as np
from scipy.linalg import lu
from numpy import linalg as la


def f(x):
    return 1 / (1 + x * x)


def fourier(xis, N):
    ff = np.zeros([N, N])
    for i in range(0, N):
        for j in range(1, N + 1):
            ff[i][j - 1] = np.sin(j * np.pi * xis[i]) if j <= N / 2 else np.cos((j - N / 2) * np.pi * xis[i])
    return ff


def do_c(N):
    m = N - 1
    xis = []  # 列向量x
    fis = []  # 列向量f
    for i in range(0, N):
        xi = i / m
        fi = f(xi)
        xis.append(xi)
        fis.append(fi)

    van = np.vander(np.array(xis), increasing=True)  # 范德蒙德矩阵
    ff = fourier(xis, N)  # F矩阵
    van_p, van_l, van_u = lu(van)
    ff_p, ff_l, ff_u = lu(ff)
    van_c = np.linalg.solve(van_u, np.linalg.solve(van_l, fis))
    ff_c = np.linalg.solve(ff_u, np.linalg.solve(ff_l, fis))

    van_norm = la.norm(np.matmul(van, xis)-fis)
    ff_norm = la.norm(np.matmul(ff, xis)-fis)
    return van_c, ff_c, van_norm, ff_norm
