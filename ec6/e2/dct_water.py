from math import *
import numpy as np
import cv2 as cv

if __name__ == "__main__":
    # 强度
    alpha = 0.02
    lena = cv.imread("../../res/lena.bmp", cv.IMREAD_GRAYSCALE)
    water = cv.imread("../../res/ahu.png", cv.IMREAD_GRAYSCALE)
    cv.imshow("P1", lena)
    lena_dct = cv.dct(np.array(lena, dtype=np.float32))
    lena_dct_with_water = lena_dct + water * alpha
    lena_with_water = cv.idct(lena_dct_with_water)
    cv.imshow("P2", np.array(lena_with_water, dtype=np.uint8))
    lena_dct_with_water = cv.dct(np.array(lena_with_water, dtype=np.float32))
    water = (lena_dct_with_water - lena_dct) / alpha
    cv.imshow("P3", np.array(water, dtype=np.uint8))
    # 抗剪切攻击性能
    # 剪切攻击
    W, H = lena.shape
    for x in range(W // 10):
        for y in range(H // 10):
            lena_with_water[x, y] = 0

    cv.imshow("P4", np.array(lena_with_water, dtype=np.uint8))
    lena_dct_with_water = cv.dct(np.array(lena_with_water, dtype=np.float32))
    water = (lena_dct_with_water - lena_dct) / alpha
    # 对水印图像进行二值化，大于平均数设置为白色，小于平均数设置为黑色
    avg = np.average(water)
    W, H = lena.shape
    for x in range(W):
        for y in range(H):
            if water[x, y] > avg:
                water[x, y] = 255
            else:
                water[x, y] = 0

    cv.imshow("P5", np.array(water, dtype=np.uint8))
    cv.waitKey()
