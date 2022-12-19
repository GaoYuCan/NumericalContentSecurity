from math import log10, pow
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    carrier: np.ndarray = cv.imread('../../res/mandril.png', cv.IMREAD_COLOR)
    water: np.ndarray = cv.imread('../../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    # 显示原图
    cv.imshow("Src", carrier)
    # 嵌入水印
    carrier_water = np.array(carrier)
    W, H = water.shape
    for x in range(W):
        for y in range(H):
            carrier_water[x, y, 0] &= 0b11111000
            carrier_water[x, y, 0] |= (water[x, y] >> 5) & 7
            carrier_water[x, y, 1] &= 0b11111100
            carrier_water[x, y, 1] |= (water[x, y] >> 3) & 3
            carrier_water[x, y, 2] &= 0b11111000
            carrier_water[x, y, 2] |= water[x, y] & 7
    cv.imshow("Embedded", carrier_water)

    M, N, RGB = carrier.shape
    PSNR = 0
    for z in range(RGB):
        MSE = 0
        for x in range(M):
            for y in range(N):
                MSE += pow(int(carrier[x, y, z]) - int(carrier_water[x, y, z]), 2)
        MSE /= M * N
        PSNR += 10 * log10(255 * 255 / MSE)
    print(f"PSNR = {PSNR / RGB}")
    # 抗剪切攻击
    for x in range(W // 2):
        for y in range(H // 2):
            carrier_water[x, y, :] = 0
    # 提取水印
    water_get = np.zeros_like(water)
    for x in range(W):
        for y in range(H):
            water_get[x, y] |= (carrier_water[x, y, 0] & 7) << 5
            water_get[x, y] |= (carrier_water[x, y, 1] & 3) << 3
            water_get[x, y] |= (carrier_water[x, y, 2] & 7)
    cv.imshow("Water", water_get)
    cv.waitKey()
