import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def do3_1():
    img_src = mpimg.imread("lena.jpg")
    print(img_src.shape)

    img = img_src.reshape(256, 256 * 3)
    global U, Sigma, VT
    U, Sigma, VT = np.linalg.svd(img)
    print(U)
    print(Sigma)
    print(VT)


def reconstruct(k):
    A = np.zeros((256, 256* 3))
    for i in range(0, k):
        Ui = U[:, i].T
        Ui = Ui.reshape(256, 1)
        VTi = VT[i, :].reshape(1, 256 * 3)
        A += Sigma[i] * (Ui.dot(VTi))

    return A


def do3_2():
    fix, ax = plt.subplots(4, 2, figsize=(4, 8))

    k = 1
    for i in range(0, 4):
        for j in range(0, 2):
            k = k * 2
            img_reconstruct = reconstruct(k)
            img_reconstruct = img_reconstruct.reshape(256, 256, 3)

            ax[i][j].imshow(img_reconstruct.astype(np.uint8))
            ax[i][j].set(title=str(k))
    plt.show()


def do3_3():
    print(Sigma)
