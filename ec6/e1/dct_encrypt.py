from math import *
import numpy as np
import cv2 as cv


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


if __name__ == "__main__":
    # 轮数
    ROUND = 5
    # 密钥
    KEY_N, KEY_B = 1, 1
    lena = cv.imread("../../res/lena.bmp", cv.IMREAD_GRAYSCALE)
    cv.imshow("P1", lena)
    lena_dct = cv.dct(np.array(lena, dtype=np.float32))
    # 加密
    lena_encrypted = lena_dct
    for i in range(ROUND):
        lena_encrypted = arnold_encrypt(lena_encrypted, KEY_N, KEY_B)
    lena_encrypted = np.array(cv.idct(lena_encrypted), dtype=np.uint8)
    cv.imshow("P2", lena_encrypted)
    # 解密
    lena_decrypted = cv.dct(np.array(lena_encrypted, dtype=np.float32))
    for i in range(ROUND):
        lena_decrypted = arnold_decrypt(lena_decrypted, KEY_N, KEY_B)
    lena_decrypted = np.array(cv.idct(lena_decrypted), dtype=np.uint8)
    cv.imshow("P3", lena_decrypted)
    cv.waitKey()
