import numpy as np
import matplotlib.pyplot as plt
from part3 import fourier
from part3 import do_c


def qr_vandermonde(N):
    a = np.arange(0, 1.001, 1 / 15)

    # 得到对应的函数值，作为f向量
    b = [((1 + x ** 2) ** -1) for x in a]

    v = np.vander(a, N, increasing=True)

    q, r = np.linalg.qr(v.T @ v)

    y = np.linalg.solve(q, v.T @ b)

    c = np.linalg.solve(r, y)

    print(c)

    return c


def qr_fourier(M):
    a = np.arange(0, 1.001, 1 / 15)

    # 得到对应的函数值，作为f向量
    b = [((1 + x ** 2) ** -1) for x in a]

    f = fourier(16, M)

    q, r = np.linalg.qr(f.T @ f)

    c = np.linalg.inv(r) @ q.T @ f.T @ b
    # c = np.linalg.linalg.inv(r).dot(q.T).dot(b)

    print(c)

    return c


def do4_1():
    print("part4_1 :")
    print("vandermonde 16 X 4 :")
    qr_vandermonde(4)
    print("vandermonde 16 X 8 :")
    qr_vandermonde(8)
    print("fourier series 16 X 4 :")
    qr_fourier(4)
    print("fourier series 16 X 8 :")
    qr_fourier(8)


def do4_2():
    # a8 = np.arange(0, 1.001, 1 / 7)
    # # 得到对应的函数值，作为f向量
    # b8 = [((1 + x ** 2) ** -1) for x in a8]
    #
    # a16 = np.arange(0, 1.001, 1 / 15)
    # # 得到对应的函数值，作为f向量
    # b16 = [((1 + x ** 2) ** -1) for x in a16]

    van_c_8, ff_c_8, van_norm_8, ff_norm_8 = do_c(8)
    van_c_16, ff_c_16, van_norm_16, ff_norm_16 = do_c(16)
    # M = N = 8
    luv8 = van_c_8
    luf8 = ff_c_8

    # M = N = 16
    luv16 = van_c_16

    luf16 = ff_c_16

    # M = 16 N = 4
    qrv4 = qr_vandermonde(4)
    qrf4 = qr_fourier(4)

    # M = 16 N = 8
    qrv8 = qr_vandermonde(8)
    qrf8 = qr_fourier(8)

    def fluv(x, arr):
        i = 0
        result = 0
        for c in arr:
            result += c * (x ** i)
            i = i + 1

        return result

    def fluf(x, N, arr):
        i = 1
        result = 0
        for c in arr:
            if i <= N / 2:
                result += c * np.sin(i * np.pi * x)
                i = i + 1
            else:
                result += c * np.cos((i - N/2) * np.pi * x)
                i = i + 1
        return result

    xs = np.arange(0, 1.0001, 0.005)
    ys = [((1 + x ** 2) ** -1) for x in xs]

    luv8s = []
    for x in xs:
        luv8s.append(fluv(x, luv8))

    luf8s = []
    for x in xs:
        luf8s.append(fluf(x, 8, luf8))

    luv16s = []
    for x in xs:
        luv16s.append(fluv(x, luv16))

    luf16s = []
    for x in xs:
        luf16s.append(fluf(x, 16, luf16))

    plt.figure(figsize=(8, 12))
    plt.subplot(2, 2, 1)

    plt.plot(xs, ys, color='red', linestyle='--', alpha=0.5)
    plt.plot(xs, luv8s, color='green', linestyle='--', alpha=0.5)
    plt.plot(xs, luf8s, color='blue', linestyle='--', alpha=0.5)
    plt.title('LU M = N = 8')

    plt.subplot(2, 2, 2)
    plt.plot(xs, ys, color='red', linestyle='--', alpha=0.5)
    plt.plot(xs, luv16s, color='green', linestyle='--', alpha=0.5)
    plt.plot(xs, luf16s, color='blue', linestyle='--', alpha=0.5)
    plt.title('LU M = N = 16')

    qrv4s = []
    for x in xs:
        qrv4s.append(fluv(x, qrv4))

    qrf4s = []
    for x in xs:
        qrf4s.append(fluf(x, 4, qrf4))

    plt.subplot(2, 2, 3)
    plt.plot(xs, ys, color='red', linestyle='--', alpha=0.5)
    plt.plot(xs, qrv4s, color='green', linestyle='--', alpha=0.5)
    plt.plot(xs, qrf4s, color='blue', linestyle='--', alpha=0.5)
    plt.title('QR M = 16 N = 4')

    qrv8s = []
    for x in xs:
        qrv8s.append(fluv(x, qrv8))

    qrf8s = []

    for x in xs:
        qrf8s.append(fluf(x, 8, qrf8))

    plt.subplot(2, 2, 4)
    plt.plot(xs, ys, color='red', linestyle='--', alpha=0.5)
    plt.plot(xs, qrv8s, color='green', linestyle='--', alpha=0.5)
    plt.plot(xs, qrf8s, color='blue', linestyle='--', alpha=0.5)
    plt.title('QR M = 16 N = 8')

    plt.show()
