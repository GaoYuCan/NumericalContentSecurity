import numpy as np
from math import gcd
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


if __name__ == '__main__':
    # 轮数
    ROUND = 5
    # 密钥
    KEY_N , KEY_B = 1, 1

    # 以灰度图形式读取文件
    ahu: np.ndarray = cv.imread('../res/anda.jpeg', cv.IMREAD_GRAYSCALE)
    cv.imshow("Display Src", ahu)
    # 加密
    encrypted_ahu = ahu
    for i in range(ROUND):
        encrypted_ahu = arnold_encrypt(encrypted_ahu, KEY_N, KEY_B)
    cv.imshow("Display Encrypted", encrypted_ahu)
    # 解密
    decrypt_ahu = encrypted_ahu
    for i in range(ROUND):
        decrypt_ahu = arnold_decrypt(decrypt_ahu, KEY_N, KEY_B)
    cv.imshow("Display Decrypted", decrypt_ahu)
    cv.waitKey()
