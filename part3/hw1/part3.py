import numpy as np
from scipy.linalg import lu
from numpy import linalg as la
import matplotlib.pyplot as plt


def f(x):
    return 1 / (1 + x * x)


def vandermonde(N):
    xis = np.arange(0, 1.001, 1 / (N - 1))
    v1 = np.vander(xis, increasing=True)
    return v1


def fourier(N, M=-1):
    if M == -1:
        M = N

    m = N - 1
    xis = np.arange(0, 1.01, 1 / m)

    ff = []
    for x in xis:
        for i in range(1, M + 1):
            if i <= M / 2:
                ff.append(np.sin(np.pi * i * x))
            else:
                ff.append(np.cos(np.pi * (i - M / 2) * x))

    ff = np.array([ff])
    ff = ff.reshape(N, M)
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

    van = vandermonde(N)  # 范德蒙德矩阵
    ff = fourier(N)  # F矩阵
    van_p, van_l, van_u = lu(van)
    ff_p, ff_l, ff_u = lu(ff)
    van_c = np.linalg.solve(van_u, np.linalg.solve(van_l, fis))
    ff_c = np.linalg.solve(ff_u, np.linalg.solve(ff_l, fis))

    van_norm = la.norm(np.matmul(van, xis)-fis)
    ff_norm = la.norm(np.matmul(ff, xis)-fis)
    return van_c, ff_c, van_norm, ff_norm


def do3_1():
    van_c_8, ff_c_8, van_norm_8, ff_norm_8 = do_c(8)
    van_c_16, ff_c_16, van_norm_16, ff_norm_16 = do_c(16)
    print("part3_1 :\nwhen N = 8")
    print("van_c :", van_c_8)
    print("ff_c :", ff_c_8)
    print("van_norm :", van_norm_8)
    print("ff_norn :", ff_norm_8)
    print("\nwhen N = 16")
    print("van_c :", van_c_16)
    print("ff_c :", ff_c_16)
    print("van_norm :", van_norm_16)
    print("ff_norn :", ff_norm_16)
    print("\n")


def do3_2():
    axes = np.arange(4, 33, 4)
    print("part3_2:")
    print("axes :", axes)

    condV = []
    condF = []
    for x in axes:
        tempV = vandermonde(x-1)
        condV.append(la.cond(tempV))

        tempF = fourier(x)
        condF.append(la.cond(tempF))

    print("cond(V) :", condV)
    print("cond(F) :", condF)
    print("\n")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    plt.yscale("log")
    # condV = [np.log10(x) for x in condV]
    plt.plot(axes, condV)
    plt.xticks(axes)
    plt.xlabel('N')
    plt.ylabel('condition number')
    plt.title("Vandermonde")

    plt.subplot(1, 2, 2)

    plt.yscale("log")
    # condF = [np.log10(x) for x in condF]
    plt.plot(axes, condF)
    plt.xticks(axes)

    plt.xlabel('N')
    plt.ylabel('condition number')
    plt.title("Fourier series")

    plt.show()


def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)


def do3_3():
    # v = np.random.randn(40)
    # v = v.reshape(8, 5)
    # print(v)
    # N, isposdef (AV ), isposdef (AF ), cond(V ), cond(F).
    col = ['N', 'isposdef(Av)', 'isposdef(Af)', 'cond(V)', 'cond(F)']

    row = np.arange(4, 33, 4)

    Avs = []
    Afs = []
    condV = []
    condF = []

    maxF = 4
    maxV = 4
    for x in row:
        tempV = vandermonde(x)
        tempF = fourier(x)

        condV.append(la.cond(tempV))
        condF.append(la.cond(tempF))

        Av = tempV.T @ tempV
        Af = tempF.T @ tempF

        if (is_pos_def(Av)):
            maxV = x
        if (is_pos_def(Af)):
            maxF = x

        Avs.append(is_pos_def(Av))
        Afs.append(is_pos_def(Af))

    v = np.array([row, Avs, Afs, condV, condF])
    v = v.T

    plt.figure(figsize=(20, 8))
    tab = plt.table(cellText=v,
                    colLabels=col,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')

    tab.scale(1, 2)
    plt.axis('off')
    plt.show()

    print("part3_3 :")
    print("the largest value of N where Av is positive definite :", maxV)
    print(la.cond(vandermonde(maxV)))

    print("the largest value of N where Af is positive definite :", maxF)
    print(la.cond(fourier(maxF)))
    print("\n")

    # 这些的N所在的位置，都在趋于"稳定"
    # 结合3-2图像


def do3_4():
    Av = vandermonde(8)

    Af = fourier(8)

    Lv = np.linalg.cholesky(Av.T @ Av)
    Lf = np.linalg.cholesky(Af.T @ Af)

    # 构造一维向量，作为xi
    a = np.arange(0, 1.001, 1 / 7)

    # 得到对应的函数值，作为f向量
    b = [((1 + x ** 2) ** -1) for x in a]

    yv = np.linalg.solve(Lv, Av.T @ b)
    cv = np.linalg.solve(Lv.T, yv)

    yf = np.linalg.solve(Lf, Af.T @ b)
    cf = np.linalg.solve(Lf.T, yf)

    print("part3_4 :")
    print("vector c of vandermonde is :", cv)

    # 验证结果
    print("verify using np.allclose(V c, b) :", np.allclose(Av @ cv, b))

    residuals_v = np.array(Av @ cv - b)

    residuals_v = [x ** 2 for x in residuals_v]

    residual_v = np.sum(residuals_v)
    print("residual of vandermonde :", residual_v)

    print("vector c of fourier series is :", cf)

    # 验证结果
    print("verify using np.allclose(V c, b) :", np.allclose(Af @ cf, b))

    residuals_f = np.array(Af @ cf - b)

    residuals_f = [x ** 2 for x in residuals_f]

    residual_f = np.sum(residuals_f)
    print("residual of fourier series :", residual_f)
    print("\n")