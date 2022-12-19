from math import cos, sin
import numpy as np
import cv2 as cv

ALPHA = 200

if __name__ == '__main__':
    lena = cv.imread('../../res/lena.bmp', cv.IMREAD_GRAYSCALE)
    water = cv.imread('../../res/ahu.png', cv.IMREAD_GRAYSCALE)
    # 转换成2值图像
    water[water == 0] = 0
    water[water == 255] = 1

    lena_fft = np.fft.fft2(lena)
    # 幅度
    lena_m = np.abs(lena_fft)
    # 相位
    lena_p = np.angle(lena_fft)
    # 向幅度中嵌入
    lena_m_with_water = lena_m + water * ALPHA
    # 重构
    W, H = lena.shape
    lena_with_water = np.zeros((W, H), dtype=np.complex_)
    for x in range(W):
        for y in range(H):
            lena_with_water[x, y] = complex(real=cos(lena_p[x, y]) * lena_m_with_water[x, y],
                                            imag=sin(lena_p[x, y]) * lena_m_with_water[x, y])
    lena_with_water = np.fft.ifft2(lena_with_water)
    # 展示嵌入后
    cv.imshow("P1", np.array(np.real(lena_with_water), dtype=np.uint8))
    # 提取水印图像
    lena_with_water_fft = np.fft.fft2(lena_with_water)
    lena_with_water_fft_m = np.abs(lena_with_water_fft)
    water_re = (lena_with_water_fft_m - lena_m) // 200
    # 还原二值图像
    water_re[water_re == 1] = 255
    cv.imshow("P2", water_re)
    # 抗剪切攻击性能
    # 剪切攻击
    for x in range(W // 2):
        for y in range(H // 2):
            lena_with_water[x, y] = 0

    cv.imshow("P3", np.array(np.real(lena_with_water), dtype=np.uint8))
    # 提取水印图像
    lena_with_water_fft_attacked = np.fft.fft2(lena_with_water)
    lena_with_water_fft_attacked_m = np.abs(lena_with_water_fft)
    water_attacked_re = (lena_with_water_fft_attacked_m - lena_m) // 200
    # 还原二值图像
    water_attacked_re[water_attacked_re == 1] = 255
    cv.imshow("P4", water_attacked_re)
    cv.waitKey()
