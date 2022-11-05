from math import log10
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    lena: np.ndarray = cv.imread('../../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    water: np.ndarray = cv.imread('../../res/ahu.png', cv.IMREAD_GRAYSCALE)
    # 判断 water 是否为 二值图像
    if water[water == 255].size + water[water == 0].size != water.size:
        print("水印图像必须为二值图像!")
        exit(-1)
    # 显示原图以及水印图
    cv.imshow("S", lena)
    cv.imshow("W", water)

    # 白色 0, 黑色 1
    water[water == 0] = 1
    water[water == 255] = 0
    # 嵌入
    lena_water = np.bitwise_or(lena & (0xff - 1), water)
    # 显示嵌入后的
    cv.imshow("SW", lena_water)
    # 提取水印
    water1 = lena_water & 1
    # 还原黑白
    water1[water1 == 0] = 255
    water1[water1 == 1] = 0
    # 显示提取出的水印图
    cv.imshow("W1", water1)

    M, N = lena.shape
    MSE = 0
    for x in range(M):
        for y in range(N):
            MSE += pow(int(lena[x, y]) - int(lena_water[x, y]), 2)
    MSE /= M * N
    PSNR = 10 * log10(255 * 255 / MSE)
    print(f"PSNR = {PSNR}")
    cv.waitKey()
