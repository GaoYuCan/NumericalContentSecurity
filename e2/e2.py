import numpy as np
from math import floor
import cv2 as cv


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
    U = 3.987654321
    X0 = 0.123456789
    NOISE = 1e-7

    # 读取图像
    lena = cv.imread('../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    # 显示原图
    cv.imshow("Display Src", lena)
    # 加密
    lena_encrypted = logistic(lena, X0, U)
    # 展示加密后的图像
    cv.imshow("Display Encrypted", lena_encrypted)
    # 验证密钥的敏感性
    lena_decrypted = logistic(lena_encrypted, X0 + NOISE, U)
    cv.imshow("Display Decrypted", lena_decrypted)
    cv.waitKey()
