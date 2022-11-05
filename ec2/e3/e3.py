import numpy as np
from math import gcd, floor, pow, log2
import cv2 as cv
import matplotlib.pyplot as plt


def arnold_encrypt(src: np.ndarray, n: int, b: int):
    M, N = src.shape
    p = N // gcd(M, N)
    mat = np.array([[1, b], [n * p, 1 + b * n * p]])
    e = np.zeros_like(src)
    for x in range(M):
        for y in range(N):
            x1, y1 = np.matmul(mat, np.array([x, y]))
            e[x1 % M, y1 % N] = src[x, y]
    return e


def arnold_decrypt(e: np.ndarray, n: int, b: int):
    M, N = e.shape
    p = N // gcd(M, N)
    src = np.zeros_like(e)
    for x1 in range(M):
        for y1 in range(N):
            y = (y1 - n * p * x1) % N
            x = (x1 - b * y) % M
            src[x, y] = e[x1, y1]
    return src


def logistic(src: np.ndarray, x0: float, u: float):
    s_box = np.zeros(src.size, dtype=np.float64)
    s_box[0] = x0
    for i in range(1, src.size):
        s_box[i] = u * s_box[i - 1] * (1 - s_box[i - 1])

    s_box = s_box.reshape(src.shape)
    e = np.zeros_like(src)
    # xor 加密
    M, N = src.shape
    for x in range(M):
        for y in range(N):
            e[x, y] = (floor(s_box[x, y] * 1e14) & 0xff) ^ src[x, y]
    return e


if __name__ == '__main__':
    # arnold 轮数， 密钥
    ROUND, KEY_N, KEY_B = 3, 1, 1
    # logistic
    U, X0 = 3.987654321, 0.123456789

    # 读取原图
    lena = cv.imread('../../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    lena_encrypted = lena
    # 显示原图
    cv.imshow("Display Src", lena)
    # 加密
    for i in range(ROUND):
        lena_encrypted = arnold_encrypt(logistic(lena_encrypted, X0, U), KEY_N, KEY_B)
    # 展示加密后的图像
    cv.imshow("Display Encrypted", lena_encrypted)

    # 绘制直方图
    fig, axes = plt.subplots(ncols=2, figsize=(20, 8), dpi=100)
    axes[0].hist(lena.reshape(lena.size), 256, linewidth=0.5, edgecolor="white")
    axes[0].set_title("before encryption")
    axes[1].hist(lena_encrypted.reshape(lena_encrypted.size), 256, linewidth=0.5, edgecolor="white")
    axes[1].set_title("after encryption")
    plt.show()

    # 计算 MSE
    M, N = lena.shape
    MSE = 0
    for x in range(M):
        for y in range(N):
            MSE += pow(int(lena[x, y]) - int(lena_encrypted[x, y]), 2)
    MSE /= M * N
    print(f"MSE = {MSE}")
    # 计算 IE
    IE_SRC = 0
    IE_ENC = 0

    for i in range(256):
        ps = lena[lena == i].size / lena.size
        pe = lena_encrypted[lena_encrypted == i].size / lena_encrypted.size
        if ps != 0:
            IE_SRC -= ps * log2(ps)
        if pe != 0:
            IE_ENC -= pe * log2(pe)
    print(f"原始图像 IE = {IE_SRC}, 密文图像 IE = {IE_ENC}")
    cv.waitKey()
